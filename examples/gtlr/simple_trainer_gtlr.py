"""GTLR-GS trainer (arXiv:2603.23192): GTAS initialization + curvature-adaptive
splitting + confidence-aware metric depth regularization.

Total loss (Eq. 11):

    L = lambda_rgb * L_rgb + lambda_ssim * L_ssim + lambda_depth * L_depth

with lambda_rgb = 0.8, lambda_ssim = 0.2 (the 3DGS default, via torch.lerp) and
lambda_depth = 1. Optionally the normal-alignment loss of Eq. 7 can be enabled.

Pipeline (run from the ``examples`` directory):

    python -m gtlr.sample_points --input_ply fused.ply --output_ply init.ply
    python -m gtlr.project_depth --data_dir DATA --ply fused.ply \
        --output_dir DATA/lidar_depth
    python -m gtlr.simple_trainer_gtlr --data_dir DATA --init_ply init.ply \
        --depth_dir DATA/lidar_depth --result_dir results/gtlr

The densification strategy is :class:`gsplat.strategy.GTLRStrategy`; everything
else follows ``examples/simple_trainer.py`` (colmap Parser/Dataset, SH colors,
learning rates, eval).
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.colmap import Dataset, Parser
from geom import (
    associate_normals,
    depth_loss,
    estimate_normals,
    laplacian_confidence,
    normal_alignment_loss,
    pixel_rays,
    plane_param_signals,
    rgb_to_gray,
    unbiased_depth,
)
from sample_points import load_ply_points

from gsplat.rendering import rasterization
from gsplat.strategy import GTLRStrategy
from gsplat.strategy.gtlr import knn_indices
from gsplat.utils import normalized_quat_to_rotmat


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    # Path to the COLMAP dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/gtlr"
    # Every N images there is a test image
    test_every: int = 8
    # Normalize the world space
    normalize_world_space: bool = True

    # Sampled point cloud ply from gtlr/sample_points.py; empty uses the SfM points
    init_ply: str = ""
    # Directory of per-image LiDAR depth .npy from gtlr/project_depth.py
    depth_dir: str = ""

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000

    # Weight for SSIM loss (lambda_rgb = 1 - ssim_lambda)
    ssim_lambda: float = 0.2
    # Weight for the confidence-aware LiDAR depth loss (Eq. 10-11)
    depth_lambda: float = 1.0
    # Start the depth loss after this many steps: the unbiased plane depth
    # (Eq. 8) is meaningless while Gaussian orientations are still random.
    depth_start_iter: int = 3000
    # Enable the normal-alignment loss (Eq. 7, part of the paper's method)
    normal_loss: bool = True
    # Weight for the normal-alignment loss
    normal_lambda: float = 0.05
    # kNN for the offline LiDAR normal estimation
    normal_knn: int = 16

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # GTLR-GS densification strategy (curvature-adaptive splitting)
    strategy: GTLRStrategy = field(default_factory=GTLRStrategy)
    # Anti-aliasing in rasterization
    antialiased: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.strategy.refine_start_iter = int(self.strategy.refine_start_iter * factor)
        self.strategy.refine_stop_iter = int(self.strategy.refine_stop_iter * factor)
        self.strategy.reset_every = int(self.strategy.reset_every * factor)
        self.strategy.refine_every = int(self.strategy.refine_every * factor)
        self.strategy.total_iters = int(self.strategy.total_iters * factor)


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)
        self.cfg = cfg
        self.device = "cuda"
        os.makedirs(cfg.result_dir, exist_ok=True)

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(self.parser, split="train")
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

        self.splats, self.optimizers = self._create_splats_with_optimizers()
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification strategy (curvature-adaptive splitting, Eq. 5-6)
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state(
            scene_scale=self.scene_scale
        )

        # Offline LiDAR depth maps and (optional) normal reference
        self.depth_maps = self._load_depth_maps()
        self.normal_ref = None
        self.gaussian_normals = None
        if cfg.normal_loss:
            self.normal_ref = self._estimate_lidar_normals()
            self.gaussian_normals = associate_normals(
                self.splats["means"], *self.normal_ref
            )

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    def _init_points(self) -> Tuple[Tensor, Tensor]:
        """Points + colors from the sampled ply, or the SfM points as fallback."""
        cfg = self.cfg
        if cfg.init_ply:
            xyz, rgb = load_ply_points(cfg.init_ply)
            return torch.from_numpy(xyz).float(), torch.from_numpy(rgb).float() / 255.0
        return (
            torch.from_numpy(self.parser.points).float(),
            torch.from_numpy(self.parser.points_rgb).float() / 255.0,
        )

    def _create_splats_with_optimizers(self):
        cfg = self.cfg
        points, rgbs = self._init_points()

        # Initialize the GS size to the average distance of the 3 nearest neighbors
        # (against a random reference subsample to bound the O(chunk x R) memory
        # of knn_indices when the init cloud is large).
        ref = points
        if len(points) > 200_000:
            ref = points[torch.randperm(len(points))[:200_000]]
        knn_device = "cpu"  # keep the (possibly shared) GPU free; one-time cost
        idx = knn_indices(points.to(knn_device), ref.to(knn_device), 4)[:, 1:].cpu()
        dist_avg = (points[idx] - points[:, None]).norm(dim=-1).mean(-1)
        scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

        n = points.shape[0]
        quats = torch.rand((n, 4))
        opacities = torch.logit(torch.full((n,), cfg.init_opa))
        colors = torch.zeros((n, (cfg.sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)

        params = [
            ("means", torch.nn.Parameter(points), cfg.means_lr * self.scene_scale),
            ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
            ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
            ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
            ("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr),
            ("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr),
        ]
        splats = torch.nn.ParameterDict({k: v for k, v, _ in params}).to(self.device)
        optimizers = {
            name: torch.optim.Adam(
                [{"params": splats[name], "lr": lr, "name": name}], eps=1e-15
            )
            for name, _, lr in params
        }
        return splats, optimizers

    def _load_depth_maps(self) -> Dict[int, Tensor]:
        """Map trainset item -> preprocessed LiDAR depth tensor (CPU, moved per step).

        Maps with fewer than 500 valid pixels are dropped: with almost no LiDAR
        coverage the z-buffer noise outweighs the regularization benefit.
        """
        cfg = self.cfg
        if not cfg.depth_dir:
            return {}
        depth_maps = {}
        for item in range(len(self.trainset)):
            name = self.parser.image_names[self.trainset.indices[item]]
            stem = os.path.splitext(os.path.basename(name))[0]
            path = os.path.join(cfg.depth_dir, stem + ".npy")
            if os.path.exists(path):
                depth = torch.from_numpy(np.load(path)).float()
                if (depth > 0).sum() >= 500:
                    depth_maps[item] = depth
        print(f"Loaded {len(depth_maps)} LiDAR depth maps from {cfg.depth_dir}")
        return depth_maps

    def _estimate_lidar_normals(self) -> Tuple[Tensor, Tensor]:
        """kNN-PCA normals on the initialization point cloud (Eq. 7 reference).

        The reference cloud is capped at 100k points to bound the association
        cost when it is refreshed after every densification.
        """
        points, _ = self._init_points()
        if len(points) > 100_000:
            points = points[torch.randperm(len(points))[:100_000]]
        normals = estimate_normals(
            points, k=self.cfg.normal_knn, ref_max=self.cfg.strategy.curvature_ref_max
        )
        return points, normals

    def rasterize_splats(
        self, camtoworlds: Tensor, Ks: Tensor, width: int, height: int, **kwargs
    ) -> Tuple[Tensor, Tensor, Dict]:
        splats = self.splats
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
        return rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors,
            viewmats=torch.linalg.inv_ex(camtoworlds).inverse,
            Ks=Ks,
            width=width,
            height=height,
            rasterize_mode="antialiased" if self.cfg.antialiased else "classic",
            **kwargs,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        max_steps = cfg.max_steps
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=True, num_workers=4,
            persistent_workers=True, pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device).float() / 255.0  # [1, H, W, 3]
            height, width = pixels.shape[1:3]

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Render RGB and the PGSR plane parameters (normal + plane distance)
            # in a single rasterization pass via extra signal channels.
            item = int(data["image_id"].item())
            want_depth = (
                cfg.depth_lambda > 0
                and step >= cfg.depth_start_iter
                and item in self.depth_maps
            )
            extra = None
            if want_depth:
                viewmats = torch.linalg.inv_ex(camtoworlds).inverse
                signals = plane_param_signals(
                    self.splats["means"],
                    self.splats["quats"],
                    self.splats["scales"],
                    viewmats,
                )  # [C, N, 4]
                # Pad to 5 channels: the CUDA kernel only instantiates a fixed
                # set of channel counts (3 RGB + 5 extra = 8 is compiled).
                extra = torch.cat(
                    [signals, signals.new_zeros(signals.shape[:-1] + (1,))], -1
                )

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                extra_signals=extra,
            )
            colors = renders[..., :3]

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # L = lambda_rgb * L_rgb + lambda_ssim * L_ssim + ... (Eq. 11)
            l1loss = (colors - pixels).abs().mean()
            ssimloss = 1.0 - self.ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)

            # Confidence-aware metric depth regularization (Eq. 8-10)
            if want_depth:
                geometry = info["render_extra_signals"][..., :4]
                normal_map, distance_map = geometry[..., :3], geometry[..., 3]
                rays = pixel_rays(Ks, height, width)
                depth, ray_dot = unbiased_depth(normal_map, distance_map, rays)
                lidar_depth = self.depth_maps[item].to(device)
                weight = laplacian_confidence(rgb_to_gray(pixels[0]))
                depthloss, _ = depth_loss(
                    depth[0], lidar_depth, weight, ray_dot[0]
                )
                loss = loss + cfg.depth_lambda * depthloss

            # Optional normal alignment (Eq. 7)
            if cfg.normal_loss and self.gaussian_normals is not None:
                shortest = self.splats["scales"].argmin(-1)
                rotmats = normalized_quat_to_rotmat(
                    F.normalize(self.splats["quats"], dim=-1)
                )
                normals_gs = rotmats.gather(
                    2, shortest[:, None, None].expand(-1, 3, 1)
                ).squeeze(2)
                normalloss = normal_alignment_loss(
                    normals_gs, self.gaussian_normals.to(device)
                )
                loss = loss + cfg.normal_lambda * normalloss

            loss.backward()

            desc = f"loss={loss.item():.3f}| sh degree={sh_degree_to_use}| "
            pbar.set_description(desc)

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            n_before = len(self.splats["means"])
            self.cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=True,
            )
            # Refresh the Gaussian -> LiDAR normal association after refine.
            if (
                cfg.normal_loss
                and self.normal_ref is not None
                and len(self.splats["means"]) != n_before
            ):
                self.gaussian_normals = associate_normals(
                    self.splats["means"], *self.normal_ref
                )

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                torch.save(
                    {"step": step, "splats": self.splats.state_dict()},
                    f"{cfg.result_dir}/ckpt_{step}.pt",
                )
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

    @torch.no_grad()
    def eval(self, step: int):
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            colors = torch.clamp(colors, 0.0, 1.0)
            pixels_p = pixels.permute(0, 3, 1, 2)
            colors_p = colors.permute(0, 3, 1, 2)
            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(self.ssim(colors_p, pixels_p))
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                f"{cfg.result_dir}/val_step{step}_{i:04d}.png",
                (canvas * 255).astype(np.uint8),
            )
        stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
        print(f"Step {step}: PSNR {stats['psnr']:.3f}, SSIM {stats['ssim']:.4f}")


def main():
    cfg = tyro.cli(Config)
    cfg.adjust_steps(1.0)
    Runner(cfg).train()


if __name__ == "__main__":
    main()
