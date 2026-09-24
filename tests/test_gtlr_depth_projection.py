"""CPU tests for GTLR LiDAR depth projection and pixel conventions."""

import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))

from gtlr import geom, project_depth


def test_project_depth_uses_half_pixel_cells_and_zbuffer():
    # K=I: continuous projection (u, v) equals (x/z, y/z). Both first points
    # land in pixel [0,1)x[0,1), whose raster sample is centred at (0.5,0.5).
    # The closest camera-z must win even when processed in separate chunks.
    points = torch.tensor(
        [
            [0.75, 0.50, 1.0],
            [1.50, 1.00, 2.0],
            [1.20, 0.50, 1.0],
        ]
    )
    K = torch.eye(3)
    w2c = torch.eye(4)
    depth = project_depth.project_depth_map(
        points, K, w2c, height=2, width=2, chunk_size=1
    )
    depth_single_chunk = project_depth.project_depth_map(
        points, K, w2c, height=2, width=2, chunk_size=100
    )
    expected = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert torch.equal(depth, expected)
    assert torch.equal(depth, depth_single_chunk)


def test_project_depth_filters_invalid_and_respects_range():
    points = torch.tensor(
        [
            [0.5, 0.5, -1.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 2.0],
            [float("nan"), 0.0, 1.0],
        ]
    )
    depth = project_depth.project_depth_map(
        points,
        torch.eye(3),
        torch.eye(4),
        height=2,
        width=2,
        min_depth=1.0,
        max_depth=3.0,
    )
    assert depth[0, 0].item() == 2.0
    assert torch.isfinite(depth).all()
    assert (depth >= 0).all()


def test_pixel_rays_match_gsplat_half_pixel_centres():
    rays = geom.pixel_rays(torch.eye(3)[None], height=2, width=2)
    assert torch.equal(rays[0, 0, 0], torch.tensor([0.5, 0.5, 1.0]))
    assert torch.equal(rays[0, 1, 1], torch.tensor([1.5, 1.5, 1.0]))


def test_depth_visualization_keeps_invalid_pixels_black():
    depth = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float32)
    rgb = geom.depth_to_rgb(depth, vmax=2.0)
    assert (rgb[0, 0] == 0).all()
    assert (rgb[1, 0] == 0).all()
    assert (rgb[0, 1] != 0).any()


def test_gtlr_parser_intrinsics_follow_perspective_roi_crop():
    class ParserStub:
        params_dict = {1: np.ones(4), 2: np.ones(4), 3: np.empty(0)}
        mask_dict = {1: None, 2: np.ones((2, 2), dtype=bool), 3: None}
        roi_undist_dict = {1: (7, 11, 100, 80), 2: (5, 6, 90, 70)}
        Ks_dict = {
            1: np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]),
            2: np.array([[90.0, 0.0, 45.0], [0.0, 90.0, 35.0], [0.0, 0.0, 1.0]]),
            3: np.eye(3),
        }

    parser = ParserStub()
    project_depth.configure_gtlr_parser(parser)
    assert parser.Ks_dict[1][0, 2] == 43.0
    assert parser.Ks_dict[1][1, 2] == 29.0
    assert parser.Ks_dict[2][0, 2] == 45.0  # fisheye already shifted by Parser
    assert np.array_equal(parser.Ks_dict[3], np.eye(3))

    # The helper is intentionally idempotent because multiple GTLR components
    # can share the same Parser instance.
    project_depth.configure_gtlr_parser(parser)
    assert parser.Ks_dict[1][0, 2] == 43.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(tests)} tests passed.")
