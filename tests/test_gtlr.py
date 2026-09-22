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

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples", "gtlr"))
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
    kappa_plane = knn_curvature(plane, plane, 16)
    kappa_edge = knn_curvature(edge, edge, 16)
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


import math  # noqa: E402  (used by _make_params_optimizers)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed.")
