"""CPU-only tests for the GTLR-GS reproduction (arXiv:2603.23192).

Covers the pure-PyTorch parts: kNN curvature (Eq. 1-2), the split threshold
schedule theta(t) (Eq. 5-6), the Laplacian confidence weight (Eq. 9), the
sampling probabilities (Eq. 4), the unbiased depth of Eq. 8 against analytic
ray-plane intersections, and the curvature gate inside ``GTLRStrategy._grow_gs``.

Runs with ``pytest tests/test_gtlr.py`` or directly ``python tests/test_gtlr.py``
(pytest is not required).
"""

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples", "gtlr"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))
sys.path.insert(0, _REPO_ROOT)

from gsplat.strategy.gtlr import (
    GTLRStrategy,
    curvature_threshold,
    knn_curvature,
    knn_indices,
)

import geom
import sample_points


def _plane_points(n=15, spacing=0.1):
    g = torch.arange(n) - (n - 1) / 2.0
    x, y = torch.meshgrid(g, g, indexing="ij")
    return torch.stack([x, y, torch.zeros_like(x)], -1).reshape(-1, 3) * spacing


def _edge_points(n=15, spacing=0.1):
    # Two perpendicular half-planes meeting at the y axis.
    g = (torch.arange(n) - (n - 1) / 2.0) * spacing
    h = torch.arange(1, n + 1) * spacing
    x_pos, y1 = torch.meshgrid(h, g, indexing="ij")
    plane_a = torch.stack([x_pos, y1, torch.zeros_like(x_pos)], -1)
    z_pos, y2 = torch.meshgrid(h, g, indexing="ij")
    plane_b = torch.stack([torch.zeros_like(z_pos), y2, z_pos], -1)
    return torch.cat([plane_a.reshape(-1, 3), plane_b.reshape(-1, 3)], 0)


def test_knn_indices():
    ref = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]])
    query = torch.tensor([[0.1, 0, 0], [4.9, 5, 5]])
    idx = knn_indices(query, ref, 2)
    assert idx.shape == (2, 2)
    assert idx[0, 0].item() == 0 and idx[1, 0].item() == 3


def test_curvature_plane_vs_edge():
    plane = _plane_points()
    edge = _edge_points()
    ids_p = torch.arange(len(plane))
    ids_e = torch.arange(len(edge))
    kappa_plane = knn_curvature(plane, plane, 16, query_ids=ids_p, ref_ids=ids_p)
    kappa_edge = knn_curvature(edge, edge, 16, query_ids=ids_e, ref_ids=ids_e)
    # Planar neighborhoods are almost flat (kappa ~ 0).
    assert kappa_plane.max() < 0.01
    # Edge neighborhoods are non-planar: clearly larger curvature.
    assert kappa_edge.max() > 0.05
    assert kappa_edge.max() > 10 * kappa_plane.max()
    # Raw range is [0, 1/3] and not renormalized.
    assert (kappa_plane >= 0).all() and (kappa_edge <= 1.0 / 3.0 + 1e-6).all()


def test_curvature_threshold_schedule():
    assert curvature_threshold(0, 30_000) == 0.1
    assert abs(curvature_threshold(30_000, 30_000) - 0.3) < 1e-9
    assert abs(curvature_threshold(15_000, 30_000) - 0.2) < 1e-9
    # clamped outside [0, T]
    assert curvature_threshold(60_000, 30_000) == 0.3
    s = GTLRStrategy()
    assert s.curvature_start == 0.1 and s.curvature_end == 0.3
    assert s.curvature_knn == 16 and s.curvature_ref_max == 200_000
    assert s.total_iters == 30_000


def test_laplacian_confidence():
    flat = torch.full((16, 16), 0.5)
    w_flat = geom.laplacian_confidence(flat)
    assert torch.allclose(w_flat, torch.ones_like(w_flat))

    impulse = torch.full((16, 16), 0.5)
    impulse[8, 8] = 1.0
    w = geom.laplacian_confidence(impulse)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert w[8, 8] < 1.0  # textured/edge pixels are downweighted
    assert w[0, 0] == 1.0  # flat pixels keep full confidence


def test_sampling_probabilities():
    torch.manual_seed(0)
    kappa = torch.rand(500)
    tau = torch.rand(500)
    probs = sample_points.sampling_probabilities(kappa, tau)
    assert torch.all(probs >= 0)
    assert abs(probs.sum().item() - 1.0) < 1e-6
    expected = 0.5 * sample_points.minmax_normalize(
        kappa
    ) + 0.5 * sample_points.minmax_normalize(tau)
    expected = expected / expected.sum()
    assert torch.allclose(probs, expected)

    idx = sample_points.sample_indices(probs.numpy(), 100, seed=0)
    assert len(idx) == 100 and len(set(idx.tolist())) == 100  # without replacement


def test_unbiased_depth_matches_ray_plane():
    torch.manual_seed(0)
    height, width = 11, 13
    K = torch.tensor([[[100.0, 0.0, 6.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]]])
    rays = geom.pixel_rays(K, height, width)  # [1, H, W, 3]

    # Plane n . x = d, blended with a constant weight w: the alpha-accumulation
    # weight must cancel in the division of Eq. 8.
    n = torch.tensor([0.2, -0.1, 0.975])
    n = n / n.norm()
    d, w = 5.0, 0.7
    normal_map = (w * n).expand(1, height, width, 3)
    distance_map = torch.full((1, height, width), w * d)
    depth, denom = geom.unbiased_depth(normal_map, distance_map, rays)

    analytic = d / (n * rays).sum(-1)
    assert torch.allclose(depth, analytic, atol=1e-5)
    assert torch.allclose(denom, w * (n * rays).sum(-1), atol=1e-5)


def _make_params_optimizers(means):
    n = means.shape[0]
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means.clone()),
            "scales": torch.nn.Parameter(torch.full((n, 3), math.log(0.5))),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(torch.zeros(n)),
        }
    )
    optimizers = {
        name: torch.optim.Adam([{"params": p, "name": name}])
        for name, p in params.items()
    }
    return params, optimizers


def _grow_state(n, device="cpu"):
    state = GTLRStrategy().initialize_state(scene_scale=1.0)
    state["grad2d"] = torch.ones(n)  # all above grow_grad2d
    state["count"] = torch.ones(n)
    return state


def test_grow_gs_curvature_gate():
    torch.manual_seed(0)
    plane = _plane_points(n=12)  # kappa ~ 0, below theta(0) = 0.1
    scatter = torch.randn(150, 3) * 0.3 + torch.tensor([0.0, 0.0, 10.0])  # kappa ~ 1/3
    means = torch.cat([plane, scatter], 0)
    n_plane, n_scatter = len(plane), len(scatter)

    # The gate itself: blocks the whole plane, passes most of the scatter cloud
    # (boundary points of a sparse cloud can fall below theta).
    strategy = GTLRStrategy()
    gate = strategy.split_gate(means, torch.ones(len(means), dtype=torch.bool), step=0)
    assert gate[:n_plane].sum() == 0
    assert gate[n_plane:].float().mean() > 0.8

    # GTLR end-to-end: only high-curvature Gaussians split.
    params, optimizers = _make_params_optimizers(means)
    state = _grow_state(len(means))
    n_dupli, n_split = strategy._grow_gs(params, optimizers, state, step=0)
    assert n_dupli == 0  # all Gaussians are "large"
    assert 0 < n_split <= n_scatter
    assert len(params["means"]) == len(means) + n_split

    # DefaultStrategy (no gate) would split every high-gradient large Gaussian.
    from gsplat.strategy import DefaultStrategy

    params2, optimizers2 = _make_params_optimizers(means)
    state2 = _grow_state(len(means))
    _, n_split_default = DefaultStrategy()._grow_gs(
        params2, optimizers2, state2, step=0
    )
    assert n_split_default == len(means)

    # Late in training theta -> 0.3 blocks even the scatter cloud.
    strategy3 = GTLRStrategy()
    params3, optimizers3 = _make_params_optimizers(means)
    state3 = _grow_state(len(means))
    _, n_split_late = strategy3._grow_gs(params3, optimizers3, state3, step=30_000)
    assert n_split_late < n_scatter


def test_knn_self_exclusion():
    """R8: self matches are excluded by identity, never by blind drop-first."""
    # Points at x = 0, 1, 2, 100; ref holds ids {1, 2, 3}.
    pts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0]])
    ref_ids = torch.tensor([1, 2, 3])
    ref = pts[ref_ids]

    # Query id 1 IS in ref: self excluded, neighbors are points 2 and 100.
    idx = knn_indices(
        pts[1:2], ref, 2, query_ids=torch.tensor([1]), ref_ids=ref_ids
    )
    assert ref[idx[0]][:, 0].tolist() == [2.0, 100.0]

    # Query id 0 is NOT in ref: the true nearest neighbor (point 1) is kept.
    idx = knn_indices(
        pts[0:1], ref, 2, query_ids=torch.tensor([0]), ref_ids=ref_ids
    )
    assert ref[idx[0]][:, 0].tolist() == [1.0, 2.0]

    # Without ids there is no self exclusion at all.
    idx = knn_indices(pts[1:2], ref, 2)
    assert ref[idx[0]][0, 0].item() == 1.0

    # Duplicated coordinates: only the exact id match is excluded, the
    # coincident distinct point stays a valid neighbor.
    dup = torch.tensor([[5.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0]])
    ids = torch.arange(3)
    idx = knn_indices(dup[0:1], dup, 1, query_ids=ids[0:1], ref_ids=ids)
    assert idx[0, 0].item() == 1  # the duplicate, not the distant point 9


def test_knn_backend_full_index_space():
    """R2: knn_indices_backend always returns indices into the original cloud."""
    torch.manual_seed(0)
    points = torch.randn(300, 3)
    k = 8

    # Exact path (ref_max >= n): neighbors must match a brute-force full-cloud
    # search with self excluded.
    idx = sample_points.knn_indices_backend(points, k, ref_max=300)
    assert idx.shape == (300, k) and idx.max() < 300
    d2 = torch.cdist(points, points)
    d2.fill_diagonal_(float("inf"))
    exact = d2.topk(k, dim=-1, largest=False).indices
    for i in [0, 5, 117, 299]:
        assert idx[i].tolist() == exact[i].tolist()

    # Subsampled path (ref_max < n): indices still address the full cloud.
    idx_sub = sample_points.knn_indices_backend(points, k, ref_max=50)
    assert idx_sub.shape == (300, k) and idx_sub.max() < 300
    # and never return the query point itself
    assert (idx_sub != torch.arange(300)[:, None]).all()


def test_sampling_degenerate():
    """R6: zero-information clouds and tiny probability support cannot crash."""
    # Constant curvature and texture -> uniform fallback, no NaN.
    probs = sample_points.sampling_probabilities(
        torch.zeros(100), torch.zeros(100)
    )
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs, torch.full_like(probs, 0.01))

    # Support smaller than M: [0, 0, 1] sampled to the full set must work.
    idx = sample_points.sample_indices(np.array([0.0, 0.0, 1.0]), 3, seed=0)
    assert len(set(idx.tolist())) == 3

    # M == N works too.
    probs = sample_points.sampling_probabilities(torch.rand(10), torch.rand(10))
    idx = sample_points.sample_indices(probs.numpy(), 10, seed=0)
    assert len(set(idx.tolist())) == 10


def test_upsampling_indices_and_interpolation():
    """M > N up-samples: all points kept, extras allocated by score, placed on
    segments between each parent and one of its nearest neighbors."""
    n, m = 50, 120
    probs = np.full(n, 1.0 / n)
    idx = sample_points.sample_indices(probs, m, seed=0)
    assert len(idx) == m
    assert (idx[:n] == np.arange(n)).all()  # every original point kept
    assert idx[n:].min() >= 0 and idx[n:].max() < n  # extras are valid draws

    torch.manual_seed(0)
    points = torch.rand(n, 3) * 10
    colors = torch.rand(n, 3)
    nn_idx = sample_points.knn_indices_backend(points, k=8, ref_max=n)
    extra = torch.from_numpy(idx[n:]).long()
    new_xyz, new_rgb = sample_points.upsample_interpolate(
        points, colors, nn_idx, idx[n:], neighbor_pool=8
    )
    assert new_xyz.shape == (m - n, 3) and new_rgb.shape == (m - n, 3)
    assert torch.isfinite(new_xyz).all()

    # Each new point lies on a segment between its parent and one of the
    # parent's 8 nearest neighbors: find the best (collinear) explanation.
    parents = points[extra]
    nbrs = points[nn_idx[extra, :8]]  # [E, P, 3]
    seg = nbrs - parents[:, None]  # parent -> neighbor vectors
    rel = new_xyz - parents  # [E, 3]
    # t = projection / |seg|^2; collinear residual must vanish for some neighbor
    t = (rel[:, None] * seg).sum(-1) / seg.pow(2).sum(-1).clamp(min=1e-12)
    resid = (rel[:, None] - t[..., None] * seg).norm(dim=-1)
    best = resid.min(dim=1)
    assert (best.values < 1e-4).all()
    t_best = t[torch.arange(len(extra)), best.indices]
    assert (t_best >= -1e-4).all() and (t_best <= 1 + 1e-4).all()


def test_hybrid_sampling():
    """base_ratio splits the budget into uniform coverage + score detail."""
    n = 100
    rng = np.random.default_rng(0)
    probs = rng.random(n)
    probs /= probs.sum()

    # M <= N: exact split, all indices unique.
    base, real, dup = sample_points.hybrid_sample_indices(
        probs, 60, base_ratio=0.5, seed=0
    )
    assert len(base) == 30 and len(real) == 30 and len(dup) == 0
    assert len(set(base.tolist()) | set(real.tolist())) == 60

    # M > N: every real point is used before duplicating.
    base, real, dup = sample_points.hybrid_sample_indices(
        probs, 150, base_ratio=0.5, seed=0
    )
    assert len(base) == 75 and len(real) == 25 and len(dup) == 50
    assert len(set(base.tolist()) | set(real.tolist())) == n

    # base_ratio = 0 is the pure score sampler (paper behavior).
    idx = sample_points.sample_indices(probs, 60, seed=0)
    base, real, dup = sample_points.hybrid_sample_indices(
        probs, 60, base_ratio=0.0, seed=0
    )
    assert len(base) == 0 and np.array_equal(idx, np.concatenate([base, real, dup]))


def test_projection_similarity_invariance():
    """R1: the parser similarity transform must not change pixel projections.

    Projecting raw points with raw cameras must equal projecting transformed
    points with transformed cameras, with depths scaled by the similarity
    factor s. This is what makes a raw-frame init ply + normalize=False and a
    transformed ply + normalize=True equivalent.
    """
    from datasets.normalize import transform_cameras, transform_points

    rng = np.random.default_rng(0)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)[None].repeat(2, 0)
    c2w[0, :3, 3] = [1.0, 2.0, 3.0]
    c2w[1, :3, 3] = [-2.0, 0.5, 5.0]
    c2w[1, :3, :3] = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # yaw 90
    points = rng.normal(size=(50, 3)) @ np.diag([1.0, 1.0, 0.2]) + [0, 0, 8.0]

    # Non-trivial similarity: scale 2.5, rotation, translation.
    s = 2.5
    theta = 0.7
    T = np.eye(4)
    T[:3, :3] = s * np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0],
        ]
    )
    T[:3, 3] = [3.0, -2.0, 1.0]

    def project(c2w_mats, pts):
        w2c = np.linalg.inv(c2w_mats)
        cam = np.einsum("nij,mj->nmi", w2c[:, :3, :3], pts) + w2c[:, None, :3, 3]
        uv = cam[..., :2] / cam[..., 2:3]
        return uv, cam[..., 2]

    uv_raw, z_raw = project(c2w, points)
    uv_norm, z_norm = project(
        transform_cameras(T, c2w.copy()), transform_points(T, points)
    )
    assert np.allclose(uv_raw, uv_norm, atol=1e-4)
    assert np.allclose(z_norm, s * z_raw, atol=1e-4)


def test_depth_map_filename():
    """R13: images with colliding basenames map to distinct depth files."""
    assert geom.depth_map_filename("0_00_00000426.jpg") == "0_00_00000426.npy"
    a = geom.depth_map_filename("cam0/0001.jpg")
    b = geom.depth_map_filename("cam1/0001.jpg")
    assert a != b and a == "cam0__0001.npy" and b == "cam1__0001.npy"


import math  # noqa: E402  (used by _make_params_optimizers)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed.")
