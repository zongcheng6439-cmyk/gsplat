"""Offline LiDAR depth maps for the GTLR-GS depth regularization (arXiv:2603.23192).

Projects the registered LiDAR point cloud into the exact pinhole image domain
used by ``datasets.colmap.Dataset``: the Parser's adjusted/undistorted K, ROI and
image size are used for both depth and visualization. Points are assigned to
the rasterizer pixel whose centre is ``(x + 0.5, y + 0.5)`` (therefore
``floor(u), floor(v)``), and the closest camera-z is kept with a z-buffer.

Each image produces one float32 ``.npy`` map. Zero marks pixels without a LiDAR
return. A manifest records the coordinate frame, factor, pixel convention,
per-image shape and coverage so the trainer can reject mismatched artifacts.

Run from the ``examples`` directory so ``datasets`` and ``sample_points`` are
importable:

    python -m gtlr.project_depth --data_dir /path/to/colmap --ply fused.ply \
        --output_dir /path/to/colmap/lidar_depth [--factor 4]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

try:
    from sample_points import load_ply_points
except ImportError:  # run as `python -m gtlr.project_depth` from examples/
    from gtlr.sample_points import load_ply_points


def project_depth_map(
    points: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    height: int,
    width: int,
    *,
    chunk_size: int = 1_000_000,
    min_depth: float = 0.0,
    max_depth: float = float("inf"),
) -> torch.Tensor:
    """Z-buffered projection of a point cloud into one camera.

    Args:
        points: [N, 3] world points. K: [3, 3] intrinsics.
        w2c: [4, 4] world-to-camera. height/width: image size.
        chunk_size: Maximum number of points projected at once. The z-buffer is
            shared across chunks, so chunking does not change the result.
        min_depth/max_depth: Accepted camera-z range, in the point-cloud frame's
            units. ``min_depth`` is exclusive and ``max_depth`` is inclusive.

    Returns:
        [H, W] float32 z-depth; 0 where no point projects (invalid).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    if K.shape != (3, 3) or w2c.shape != (4, 4):
        raise ValueError(f"expected K [3,3] and w2c [4,4], got {K.shape}, {w2c.shape}")
    if height <= 0 or width <= 0 or chunk_size <= 0:
        raise ValueError("height, width and chunk_size must be positive")
    if min_depth < 0 or max_depth <= min_depth:
        raise ValueError("require 0 <= min_depth < max_depth")

    device = points.device
    K = K.to(device=device, dtype=points.dtype)
    w2c = w2c.to(device=device, dtype=points.dtype)
    R, t = w2c[:3, :3], w2c[:3, 3]
    depth = torch.full(
        (height * width,), float("inf"), dtype=points.dtype, device=device
    )

    for world in points.split(chunk_size):
        cam = world @ R.T + t
        z = cam[:, 2]
        front = (
            torch.isfinite(cam).all(dim=-1)
            & (z > min_depth)
            & (z <= max_depth)
        )
        if not front.any():
            continue

        cam = cam[front]
        z = z[front]
        proj = cam @ K.T
        uv = proj[:, :2] / proj[:, 2:3]
        finite = torch.isfinite(uv).all(dim=-1)
        if not finite.any():
            continue
        uv = uv[finite]
        z = z[finite]

        # gsplat samples pixel (x, y) at (x + 0.5, y + 0.5). The matching
        # pixel cell is [x, x + 1) x [y, y + 1), hence floor rather than round.
        u = torch.floor(uv[:, 0]).long()
        v = torch.floor(uv[:, 1]).long()
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not inside.any():
            continue
        flat = v[inside] * width + u[inside]
        depth.scatter_reduce_(0, flat, z[inside], reduce="amin", include_self=True)

    depth = depth.reshape(height, width)
    return torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)


def load_parser_image(parser, index: int) -> np.ndarray:
    """Load the RGB image exactly as ``datasets.colmap.Dataset`` does.

    ``Parser.Ks_dict`` and ``imsize_dict`` describe an undistorted, and for
    fisheye cameras cropped, pinhole image. Reading ``parser.image_paths``
    directly would overlay such a depth map on the original distorted image.
    """
    import imageio.v2 as imageio

    image = imageio.imread(parser.image_paths[index])[..., :3]
    camera_id = parser.camera_ids[index]
    params = parser.params_dict[camera_id]
    if len(params) > 0:
        import cv2

        image = cv2.remap(
            image,
            parser.mapx_dict[camera_id],
            parser.mapy_dict[camera_id],
            cv2.INTER_LINEAR,
        )
        x, y, w, h = parser.roi_undist_dict[camera_id]
        image = image[y : y + h, x : x + w]

    expected_width, expected_height = parser.imsize_dict[camera_id]
    if image.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            f"preprocessed image {parser.image_names[index]} has shape "
            f"{image.shape[:2]}, expected {(expected_height, expected_width)}"
        )
    return np.ascontiguousarray(image)


def configure_gtlr_parser(parser) -> None:
    """Align Parser intrinsics with its cropped training images, once.

    ``datasets.colmap.Parser`` subtracts the crop origin for fisheye cameras,
    but its perspective-distortion branch stores OpenCV's full undistorted K
    and then crops the image to ``roi_undist`` without shifting the principal
    point. GTLR consumes that K in three places (LiDAR projection, RGB training
    and validation), so apply the missing ROI shift consistently here.
    """
    if getattr(parser, "_gtlr_intrinsics_configured", False):
        return
    for camera_id, params in parser.params_dict.items():
        if len(params) == 0:
            continue
        # A non-None mask identifies Parser's fisheye branch, where the shift
        # has already been applied. The perspective branch keeps mask=None.
        if parser.mask_dict[camera_id] is not None:
            continue
        x, y, _, _ = parser.roi_undist_dict[camera_id]
        K = parser.Ks_dict[camera_id].copy()
        K[0, 2] -= x
        K[1, 2] -= y
        parser.Ks_dict[camera_id] = K
    parser._gtlr_intrinsics_configured = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True, help="COLMAP dataset directory")
    parser.add_argument("--ply", required=True, help="Registered LiDAR point cloud ply")
    parser.add_argument("--output_dir", required=True, help="Where to write .npy depth maps")
    parser.add_argument("--factor", type=int, default=4, help="Image downsample factor")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Projection device (default: CUDA when available)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1_000_000,
        help="Maximum LiDAR points projected at once",
    )
    parser.add_argument("--min_depth", type=float, default=0.0)
    parser.add_argument("--max_depth", type=float, default=float("inf"))
    parser.add_argument(
        "--vis_every",
        type=int,
        default=100,
        help="Save a depth visualization PNG every N images (0 disables)",
    )
    args = parser.parse_args()

    from datasets.colmap import Parser

    try:
        from gtlr.geom import depth_map_filename, depth_to_rgb
    except ImportError:  # run as a plain script from the gtlr directory
        from geom import depth_map_filename, depth_to_rgb

    # GTLR uses registered LiDAR depth as a metric-scale anchor. Keep LiDAR,
    # cameras and Gaussians in the original COLMAP/world frame throughout.
    colmap = Parser(data_dir=args.data_dir, factor=args.factor, normalize=False)
    configure_gtlr_parser(colmap)
    xyz, _ = load_ply_points(args.ply)
    device = resolve_device(args.device)
    points = torch.from_numpy(np.ascontiguousarray(xyz)).float().to(device)
    print(
        f"Projecting {len(points):,} points into {len(colmap.image_names)} images "
        f"on {device} (factor={args.factor}, raw metric frame)"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.vis_every > 0:
        os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)
    files = []
    file_set = set()
    image_stats = {}
    for i, name in enumerate(colmap.image_names):
        camera_id = colmap.camera_ids[i]
        K = torch.from_numpy(colmap.Ks_dict[camera_id]).float().to(device)
        width, height = colmap.imsize_dict[camera_id]
        w2c = (
            torch.from_numpy(np.linalg.inv(colmap.camtoworlds[i])).float().to(device)
        )
        depth = project_depth_map(
            points,
            K,
            w2c,
            height,
            width,
            chunk_size=args.chunk_size,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
        ).cpu()
        fname = depth_map_filename(name)
        if fname in file_set:
            raise ValueError(f"depth map name collision for image {name}")
        files.append(fname)
        file_set.add(fname)
        np.save(os.path.join(args.output_dir, fname), depth.numpy())
        valid = depth > 0
        valid_count = int(valid.sum().item())
        if valid_count:
            valid_depth = depth[valid]
            min_valid = float(valid_depth.min().item())
            max_valid = float(valid_depth.max().item())
        else:
            min_valid = max_valid = None
        image_stats[name] = {
            "file": fname,
            "height": height,
            "width": width,
            "valid_pixels": valid_count,
            "min_depth": min_valid,
            "max_depth": max_valid,
        }
        print(
            f"[{i + 1}/{len(colmap.image_names)}] {name}: "
            f"{valid_count}/{height * width} valid px"
        )

        if args.vis_every > 0 and i % args.vis_every == 0:
            import imageio.v2 as imageio

            depth_np = depth.numpy()
            valid_np = depth_np > 0
            if valid_np.any():
                vmax = float(np.quantile(depth_np[valid_np], 0.95))
            else:
                vmax = 1.0
            depth_vis = depth_to_rgb(depth_np, vmax)
            image = load_parser_image(colmap, i).astype(np.float32) / 255.0
            overlay = (image * 128).astype(np.uint8)
            overlay[valid_np] = (
                0.4 * overlay[valid_np] + 0.6 * depth_vis[valid_np]
            ).astype(np.uint8)
            canvas = np.concatenate([overlay, depth_vis], axis=1)
            stem = os.path.splitext(os.path.basename(fname))[0]
            imageio.imwrite(
                os.path.join(args.output_dir, "vis", f"{stem}.png"), canvas
            )

    manifest = {
        "version": 2,
        "factor": args.factor,
        "normalize": False,
        "coordinate_frame": "raw_metric",
        "depth_type": "camera_z",
        "pixel_convention": "half_pixel_centers_floor",
        "cropped_intrinsics_aligned": True,
        "source_ply": os.path.basename(args.ply),
        "num_points": int(points.shape[0]),
        "min_depth": args.min_depth,
        "max_depth": None if not np.isfinite(args.max_depth) else args.max_depth,
        "images": dict(zip(colmap.image_names, files)),
        "image_stats": image_stats,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
