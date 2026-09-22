"""Offline LiDAR depth maps for the GTLR-GS depth regularization (arXiv:2603.23192).

Projects the registered LiDAR point cloud into every training camera using the
COLMAP intrinsics/extrinsics, keeps the closest z per pixel with a z-buffer
(torch scatter_reduce amin) and stores one ``<image_stem>.npy`` float32 depth
map per image (0 marks pixels without any LiDAR return; the valid-pixel set U
of Eq. 10 is ``depth > 0``).

Run from the ``examples`` directory so ``datasets`` and ``sample_points`` are
importable:

    python -m gtlr.project_depth --data_dir /path/to/colmap --ply fused.ply \
        --output_dir /path/to/colmap/lidar_depth [--factor 4]
"""

from __future__ import annotations

import argparse
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
) -> torch.Tensor:
    """Z-buffered projection of a point cloud into one camera.

    Args:
        points: [N, 3] world points. K: [3, 3] intrinsics.
        w2c: [4, 4] world-to-camera. height/width: image size.

    Returns:
        [H, W] float32 z-depth; 0 where no point projects (invalid).
    """
    R, t = w2c[:3, :3], w2c[:3, 3]
    cam = points @ R.T + t  # [N, 3]
    z = cam[:, 2]
    proj = cam @ K.T
    u = (proj[:, 0] / z).round().long()
    v = (proj[:, 1] / z).round().long()
    inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    flat = v[inside] * width + u[inside]
    depth = torch.full((height * width,), float("inf"), device=points.device)
    depth.scatter_reduce_(0, flat, z[inside], reduce="amin")
    depth = depth.reshape(height, width)
    return torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True, help="COLMAP dataset directory")
    parser.add_argument("--ply", required=True, help="Registered LiDAR point cloud ply")
    parser.add_argument("--output_dir", required=True, help="Where to write .npy depth maps")
    parser.add_argument("--factor", type=int, default=4, help="Image downsample factor")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply the Parser world normalization so depths match a trainer "
        "running with normalize_world_space=True",
    )
    args = parser.parse_args()

    from datasets.colmap import Parser
    from datasets.normalize import transform_points

    colmap = Parser(data_dir=args.data_dir, factor=args.factor, normalize=args.normalize)
    xyz, _ = load_ply_points(args.ply)
    if args.normalize:
        xyz = transform_points(colmap.transform, xyz)
    points = torch.from_numpy(np.ascontiguousarray(xyz)).float()

    os.makedirs(args.output_dir, exist_ok=True)
    for i, name in enumerate(colmap.image_names):
        camera_id = colmap.camera_ids[i]
        K = torch.from_numpy(colmap.Ks_dict[camera_id]).float()
        width, height = colmap.imsize_dict[camera_id]
        w2c = torch.from_numpy(np.linalg.inv(colmap.camtoworlds[i])).float()
        depth = project_depth_map(points, K, w2c, height, width)
        stem = os.path.splitext(os.path.basename(name))[0]
        np.save(os.path.join(args.output_dir, stem + ".npy"), depth.numpy())
        print(f"[{i + 1}/{len(colmap.image_names)}] {name}: {(depth > 0).sum().item()} valid px")


if __name__ == "__main__":
    main()
