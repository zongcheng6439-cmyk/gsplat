"""GTLR-GS confidence-aware metric depth regularization (arXiv:2603.23192, Eq. 7-10).

Unbiased depth rendering follows the PGSR-style plane-parameter blending used
in ``gsplat_remote/fresh/pgsr_geom.py``: each Gaussian is flattened along its
shortest-scale axis into a local plane (normal n_i, camera-to-plane distance
D_i), the 4 per-Gaussian channels are alpha-accumulated through gsplat's
``extra_signals``, and the unbiased depth of Eq. 8 is

    D^(p) = (sum_i w_i D_i) / ((sum_i w_i n_i) . K^-1 p~),

where the blending weights cancel in the division, leaving the intersection of
the pixel ray with the blended local plane (a camera z-depth, since rays have
z = 1). Pixels with a near-grazing denominator are clamped and gated.

The depth loss of Eq. 10 is a confidence-weighted L1 against the z-buffered
LiDAR depth on the valid pixel set U, with the Laplacian confidence of Eq. 9

    w(p) = 1 - |nabla^2 I(p)| / max_q |nabla^2 I(q)|.

The optional normal-alignment loss of Eq. 7,
mean(1 - |n_gs . n_lidar|), uses LiDAR normals estimated offline by kNN PCA and
associated to Gaussians by nearest center.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.rendering import rasterization
from gsplat.utils import normalized_quat_to_rotmat


def rgb_to_gray(image: Tensor) -> Tensor:
    return (image * image.new_tensor([0.299, 0.587, 0.114])).sum(-1)


def depth_map_filename(image_name: str) -> str:
    """Depth map file name for a COLMAP image name.

    Encodes the full relative path (`cam0/0001.jpg` -> `cam0__0001.npy`) so
    images with colliding basenames from different cameras cannot overwrite
    each other. Flat names keep their plain stem.
    """
    stem = os.path.splitext(image_name)[0]
    return stem.replace(os.sep, "__").replace("/", "__") + ".npy"


def _safe_denominator(value: Tensor) -> Tensor:
    return torch.where(
        value.abs() < 1e-6,
        value.sign().masked_fill(value == 0, 1.0) * 1e-6,
        value,
    )


def plane_param_signals(
    means: Tensor, quats: Tensor, scales_log: Tensor, viewmats: Tensor
) -> Tensor:
    """Per-Gaussian plane parameters in each camera frame.

    Normal = shortest-scale axis transformed to the camera frame and flipped to
    face the camera; distance = n . mu_cam, the signed camera-to-plane distance.

    Args:
        means: [N, 3], quats: [N, 4], scales_log: [N, 3] (log space; argmin is
            unaffected by the exp activation), viewmats: [C, 4, 4] world-to-cam.

    Returns:
        [C, N, 4] extra signals: (nx, ny, nz, distance).
    """
    rotations = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))
    shortest = scales_log.argmin(-1)
    normals_world = rotations.gather(
        2, shortest[:, None, None].expand(-1, 3, 1)
    ).squeeze(2)
    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    centers_cam = torch.einsum("cij,nj->cni", R, means) + t[:, None]
    normals_cam = torch.einsum("cij,nj->cni", R, normals_world)
    facing_away = (normals_cam * centers_cam).sum(-1, keepdim=True) > 0
    normals_cam = torch.where(facing_away, -normals_cam, normals_cam)
    distances = (normals_cam * centers_cam).sum(-1, keepdim=True)
    return torch.cat([normals_cam, distances], dim=-1)


def pixel_rays(Ks: Tensor, height: int, width: int) -> Tensor:
    """Unit-z camera rays [C, H, W, 3] at integer pixel coordinates (K^-1 p~)."""
    dtype, device = Ks.dtype, Ks.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    return torch.stack(
        (
            (x - Ks[:, 0, 2, None, None]) / Ks[:, 0, 0, None, None],
            (y - Ks[:, 1, 2, None, None]) / Ks[:, 1, 1, None, None],
            ones.expand(Ks.shape[0], -1, -1),
        ),
        dim=-1,
    )


def unbiased_depth(
    normal_map: Tensor, distance_map: Tensor, rays: Tensor
) -> tuple[Tensor, Tensor]:
    """Eq. 8: D(p) = D_blended / (N(p) . K^-1 p~).

    Returns (depth, raw denominator); callers should gate pixels whose
    |denominator| is tiny (grazing angles or inconsistent blended normals).
    """
    denominator = (normal_map * rays).sum(-1)
    return distance_map / _safe_denominator(denominator), denominator


@dataclass
class PlaneRender:
    normal_map: Tensor  # [C, H, W, 3] blended plane normals (unnormalized)
    distance_map: Tensor  # [C, H, W] blended camera-to-plane distances
    depth: Tensor  # [C, H, W] unbiased ray-plane z-depth
    ray_dot: Tensor  # [C, H, W] raw N . ray denominator before clamping
    rays: Tensor  # [C, H, W, 3]
    alpha: Tensor  # [C, H, W]


def render_plane_params(
    means: Tensor,
    quats: Tensor,
    scales_log: Tensor,
    opacities_log: Tensor,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    *,
    rasterize_mode: str = "classic",
    camera_model: str = "pinhole",
) -> PlaneRender:
    """Render the plane maps of Eq. 8 in one pass through ``extra_signals``.

    Gradients flow from the rendered maps back through the extra channels to
    quats (via normals) and to means/quats (via distances). Dummy zero colors
    keep the standard RGB kernel path; the CUDA kernel only instantiates a
    fixed set of channel counts, so the 4 geometry channels are padded to 5
    (3 colors + 5 extra = 8 channels, which is compiled; 7 is not).
    """
    viewmats = torch.linalg.inv_ex(camtoworlds).inverse
    signals = plane_param_signals(means, quats, scales_log, viewmats)
    extra = torch.cat([signals, signals.new_zeros(signals.shape[:-1] + (1,))], -1)
    colors = means.new_zeros((signals.shape[0], means.shape[0], 3))
    _, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales_log.exp(),
        opacities=opacities_log.sigmoid(),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB",
        packed=True,
        rasterize_mode=rasterize_mode,
        camera_model=camera_model,
        extra_signals=extra,
    )
    geometry = info["render_extra_signals"][..., :4]
    normal_map, distance_map = geometry[..., :3], geometry[..., 3]
    rays = pixel_rays(Ks, height, width)
    depth, ray_dot = unbiased_depth(normal_map, distance_map, rays)
    return PlaneRender(
        normal_map=normal_map,
        distance_map=distance_map,
        depth=depth,
        ray_dot=ray_dot,
        rays=rays,
        alpha=alphas[..., 0],
    )


def laplacian_confidence(gray: Tensor) -> Tensor:
    """Eq. 9: w(p) = 1 - |nabla^2 I(p)| / max_q |nabla^2 I(q)|.

    Args:
        gray: [H, W] GT gray image.

    Returns:
        [H, W] confidence weights in [0, 1]; flat regions have w = 1.
    """
    kernel = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(padded, kernel[None, None])[0, 0].abs()
    lap_max = lap.max()
    if lap_max < 1e-12:
        return torch.ones_like(gray)
    return (1.0 - lap / lap_max).clamp(0.0, 1.0)


def depth_loss(
    depth: Tensor,
    lidar_depth: Tensor,
    weight: Tensor,
    ray_dot: Tensor,
    *,
    min_ray_dot: float = 0.1,
    max_err: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Eq. 10: L_depth = (1/|U|) sum_{p in U} w(p) |D_LiDAR(p) - D^(p)|.

    Args:
        depth: [H, W] unbiased rendered z-depth (carries gradients).
        lidar_depth: [H, W] z-buffered LiDAR z-depth; 0 marks invalid pixels.
        weight: [H, W] confidence weights of Eq. 9.
        ray_dot: [H, W] raw N . ray denominator, used to gate grazing angles.
        min_ray_dot: gate on |N . ray|; below it the blended planes are nearly
            edge-on and the unbiased depth explodes.
        max_err: per-pixel clamp on the absolute depth error (normalized scene
            units), keeping single-view outliers from dominating the loss.

    Returns:
        (loss, valid mask U). The loss is 0 (with gradient path) if U is empty.
    """
    valid = (
        (lidar_depth > 0)
        & torch.isfinite(depth)
        & (depth > 0)
        & (ray_dot.abs() > min_ray_dot)
    )
    if not valid.any():
        return depth.reshape(-1)[:0].sum(), valid
    err = (lidar_depth - depth).abs().clamp(max=max_err)
    return (weight * err)[valid].mean(), valid


def normal_alignment_loss(normals_gs: Tensor, normals_lidar: Tensor) -> Tensor:
    """Eq. 7: L_normal = mean(1 - |n_gs . n_lidar|), sign-invariant."""
    dots = (F.normalize(normals_gs, dim=-1) * F.normalize(normals_lidar, dim=-1)).sum(-1)
    return (1.0 - dots.abs()).mean()


def estimate_normals(points: Tensor, k: int = 16, ref_max: int = 200_000) -> Tensor:
    """kNN-PCA normals of a point cloud (smallest-eigenvector direction).

    The neighbor reference set is randomly subsampled to at most ``ref_max``
    points, same approximation as the online curvature of the strategy. Self
    matches are excluded by identity via query_ids/ref_ids.
    """
    from gsplat.strategy.gtlr import knn_indices

    n = points.shape[0]
    ref_ids = torch.arange(n, device=points.device)
    if n > ref_max:
        ref_ids = torch.randperm(n, device=points.device)[:ref_max]
    idx = knn_indices(
        points,
        points[ref_ids],
        k,
        query_ids=torch.arange(n, device=points.device),
        ref_ids=ref_ids,
    )
    nbrs = points[ref_ids][idx]
    centered = nbrs - nbrs.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / max(idx.shape[1] - 1, 1)
    eigvecs = torch.linalg.eigh(cov).eigenvectors  # ascending eigenvalues
    return eigvecs[..., 0]


def associate_normals(means: Tensor, ref_points: Tensor, ref_normals: Tensor) -> Tensor:
    """Nearest-center association of precomputed LiDAR normals to Gaussians.

    Runs on CPU (returns a CPU tensor): during training the GPU is busy with
    the rasterizer, and the chunked kNN transient would not fit next to it.
    """
    from gsplat.strategy.gtlr import knn_indices

    idx = knn_indices(means.detach().cpu(), ref_points.cpu(), 1)[:, 0]
    return ref_normals.cpu()[idx]
