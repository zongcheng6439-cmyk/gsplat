"""Offline validation of the GTLR-GS pipeline on real data (no training).

Checks, using an existing checkpoint and the preprocessed artifacts:

1. LiDAR depth map alignment: the projected depth maps are overlaid on the
   corresponding training images (catches pose / coordinate-frame mismatch).
2. Unbiased depth (Eq. 8) end-to-end: rendered unbiased depth is compared
   against the LiDAR depth maps with the confidence weights and the gates of
   ``depth_loss`` (Eq. 9-10); per-view error statistics are reported.
3. Sampling quality (Eq. 1-4): curvature and texture complexity of the sampled
   initialization cloud vs. a random subset of the full cloud.

Run from the ``examples`` directory:

    python -m gtlr.validate --data_dir DATA --ckpt results/smoke/ckpt_999.pt \
        --depth_dir DATA/lidar_depth --full_ply fused.ply \
        --sampled_ply init.ply --output_dir results/validate
"""

from __future__ import annotations

import argparse
import os
import sys

import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.colmap import Parser

from geom import (
    depth_loss,
    depth_map_filename,
    laplacian_confidence,
    pixel_rays,
    plane_param_signals,
    rgb_to_gray,
    unbiased_depth,
)
from sample_points import curvature_texture, load_ply_points

from gsplat.rendering import rasterization


def _depth_to_rgb(depth: np.ndarray, vmax: float) -> np.ndarray:
    """Simple turbo-ish blue->red colormap for a depth map (0 = invalid)."""
    t = np.clip(depth / max(vmax, 1e-6), 0.0, 1.0)
    r = np.clip(1.5 * t - 0.25, 0, 1)
    g = np.clip(1.5 - np.abs(2 * t - 1.0) * 1.5, 0, 1)
    b = np.clip(1.25 - 1.5 * t, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


@torch.no_grad()
def validate_depth(
    ckpt_path: str,
    parser: Parser,
    depth_dir: str,
    output_dir: str,
    n_views: int = 20,
    n_vis: int = 4,
) -> None:
    """Render unbiased depth for a few views and compare with LiDAR depth."""
    device = "cuda"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    print(f"checkpoint step {ckpt['step']}, {len(splats['means'])} gaussians")

    indices = np.linspace(0, len(parser.image_names) - 1, n_views).astype(int)
    stats = []
    vis_count = 0
    for i in indices:
        name = parser.image_names[i]
        stem = os.path.splitext(os.path.basename(name))[0]
        depth_path = os.path.join(depth_dir, depth_map_filename(name))
        if not os.path.exists(depth_path):
            continue
        lidar_depth = torch.from_numpy(np.load(depth_path)).float().to(device)

        camtoworlds = torch.from_numpy(parser.camtoworlds[i : i + 1]).float().to(device)
        camera_id = parser.camera_ids[i]
        Ks = torch.from_numpy(parser.Ks_dict[camera_id]).float().to(device)[None]
        width, height = parser.imsize_dict[camera_id]
        viewmats = torch.linalg.inv_ex(camtoworlds).inverse

        signals = plane_param_signals(
            splats["means"], splats["quats"], splats["scales"], viewmats
        )
        extra = torch.cat([signals, signals.new_zeros(signals.shape[:-1] + (1,))], -1)
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        renders, alphas, info = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=0,
            render_mode="RGB",
            packed=True,
            extra_signals=extra,
        )
        geometry = info["render_extra_signals"][..., :4]
        rays = pixel_rays(Ks, height, width)
        depth, ray_dot = unbiased_depth(geometry[..., :3], geometry[..., 3], rays)

        image = (
            imageio.imread(parser.image_paths[i]).astype(np.float32) / 255.0
        )
        weight = laplacian_confidence(
            rgb_to_gray(torch.from_numpy(image).to(device))
        )
        loss, valid = depth_loss(depth[0], lidar_depth, weight, ray_dot[0])
        err = (lidar_depth - depth[0]).abs()[valid]
        lidar_valid = lidar_depth[lidar_depth > 0]
        if err.numel() == 0:
            print(f"{name}: no valid pixels after gating, skipped")
            continue
        med = err.median().item()
        p90 = err.quantile(0.9).item()
        within = (err < 0.05).float().mean().item()
        stats.append((med, p90, within))
        print(
            f"{name}: valid {valid.sum().item()}/{int((lidar_depth > 0).sum())}, "
            f"loss {loss.item():.4f}, median |err| {med:.4f}, "
            f"p90 {p90:.4f}, frac<0.05 {within:.3f}, "
            f"lidar depth med {lidar_valid.median().item():.3f}"
        )

        if vis_count < n_vis:
            vis_count += 1
            rendered = (renders[0, ..., :3].clamp(0, 1).cpu().numpy() * 255)
            vmax = float(lidar_valid.quantile(0.95).item())
            lidar_vis = _depth_to_rgb(lidar_depth.cpu().numpy(), vmax)
            render_vis = _depth_to_rgb(
                np.where(valid.cpu().numpy(), depth[0].cpu().numpy(), 0.0), vmax
            )
            # overlay: image dimmed + valid lidar pixels in color
            overlay = (image * 128).astype(np.uint8)
            lidar_mask = lidar_depth.cpu().numpy() > 0
            overlay[lidar_mask] = (
                0.4 * overlay[lidar_mask] + 0.6 * lidar_vis[lidar_mask]
            ).astype(np.uint8)
            canvas = np.concatenate(
                [overlay, rendered.astype(np.uint8), lidar_vis, render_vis], axis=1
            )
            imageio.imwrite(
                os.path.join(output_dir, f"depth_check_{stem}.png"), canvas
            )

    if stats:
        arr = np.array(stats)
        print(
            f"\n== {len(stats)} views: median |err| {arr[:, 0].mean():.4f}, "
            f"p90 {arr[:, 1].mean():.4f}, frac<0.05 {arr[:, 2].mean():.3f}"
        )


def validate_sampling(full_ply: str, sampled_ply: str, n: int = 100_000) -> None:
    """Compare curvature/texture of the sampled init cloud vs random baseline.

    Complexity is computed once on the FULL cloud (kNN against a fixed 200k
    reference subsample, same as sample_points.py), then the sampled points are
    matched back to their exact indices so the comparison shares the same
    normalization and neighborhood density.
    """
    from scipy.spatial import cKDTree

    xyz_full, rgb_full = load_ply_points(full_ply)
    xyz_s, _ = load_ply_points(sampled_ply)

    points = torch.from_numpy(np.ascontiguousarray(xyz_full)).float()
    colors = torch.from_numpy(np.ascontiguousarray(rgb_full)).float() / 255.0
    kappa, tau = curvature_texture(points, colors, k=64)
    kappa_n = (kappa - kappa.min()) / (kappa.max() - kappa.min())
    tau_n = (tau - tau.min()) / (tau.max() - tau.min())

    dist, idx_s = cKDTree(xyz_full).query(xyz_s)
    assert dist.max() == 0.0, "sampled points are not a subset of the full cloud"
    rng = np.random.default_rng(0)
    idx_r = rng.choice(len(xyz_full), len(xyz_s), replace=False)

    for name, vals in [("kappa", kappa_n.numpy()), ("tau", tau_n.numpy())]:
        s_mean, r_mean = vals[idx_s].mean(), vals[idx_r].mean()
        print(
            f"{name}: sampled mean {s_mean:.4f} | random mean {r_mean:.4f} | "
            f"enrichment {s_mean / max(r_mean, 1e-12):.2f}x"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", default="", help="Checkpoint for depth validation")
    parser.add_argument("--depth_dir", default="")
    parser.add_argument("--full_ply", default="")
    parser.add_argument("--sampled_ply", default="")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Use the normalized parser frame (must match the trained model)",
    )
    parser.add_argument("--n_views", type=int, default=20)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.ckpt and args.depth_dir:
        colmap = Parser(data_dir=args.data_dir, factor=args.factor, normalize=args.normalize)
        validate_depth(
            args.ckpt, colmap, args.depth_dir, args.output_dir, n_views=args.n_views
        )
    if args.full_ply and args.sampled_ply:
        validate_sampling(args.full_ply, args.sampled_ply)


if __name__ == "__main__":
    main()
