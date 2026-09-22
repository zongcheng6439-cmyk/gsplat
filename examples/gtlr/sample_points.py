"""GTLR-GS geometry-texture aware sampling (arXiv:2603.23192, Eq. 1-4).

Offline Gaussian initialization: for every point p_i of the registered LiDAR
point cloud (with RGB), the kNN (k = 64) covariance eigenvalues
lambda_1 <= lambda_2 <= lambda_3 give the curvature (Eq. 1-2)

    kappa_i = lambda_1 / (lambda_1 + lambda_2 + lambda_3 + eps),

and the texture complexity is the neighborhood RGB variance (Eq. 3)

    tau_i = (1 / 3k) sum_m sum_j (c_j^(m) - c_bar_i^(m))^2.

Both are min-max normalized to [0, 1] and combined into the sampling
probability (Eq. 4, alpha = beta = 0.5)

    P_i = (0.5 * kappa^_i + 0.5 * tau^_i) / sum(...),

from which M = 3,000,000 points are drawn without replacement as the Gaussian
initialization, written back out as a gsplat-compatible xyz+RGB ply.

kNN backend: open3d when installed, otherwise chunked torch matmul topk
against a random reference subsample (same approximation as the strategy).

Usage:
    python sample_points.py --input_ply fused.ply --output_ply init_3m.ply \
        [--num_samples 3000000] [--knn 64]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import Tensor

try:
    import open3d as o3d
except ImportError:
    o3d = None

_PLY_DTYPES = {
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "char": np.int8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
}


def load_ply_points(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load xyz + RGB from a ply file; returns (float32 [N,3], uint8 [N,3]).

    Uses open3d when available, else a minimal reader supporting ascii and
    binary_little_endian vertex elements with x/y/z and red/green/blue (any
    property order, extra properties skipped).
    """
    if o3d is not None:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = (np.asarray(pcd.colors) * 255.0).round().astype(np.uint8)
        return xyz, rgb

    with open(path, "rb") as f:
        # header
        fmt, n_vert, props = None, 0, []
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n_vert = int(line.split()[2])
            elif line.startswith("element") and not line.startswith("element vertex"):
                # only the vertex element is supported before end_header
                pass
            elif line.startswith("property"):
                parts = line.split()
                props.append((parts[1], parts[2]))  # (dtype, name)
            elif line == "end_header":
                break
        names = [name for _, name in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n_vert, dtype=np.float64)
            xyz = data[:, [names.index(c) for c in ("x", "y", "z")]].astype(np.float32)
            rgb = data[:, [names.index(c) for c in ("red", "green", "blue")]].astype(
                np.uint8
            )
        elif fmt == "binary_little_endian":
            dtype = np.dtype([(name, "<" + np.dtype(_PLY_DTYPES[t]).str[1:]) for t, name in props])
            data = np.fromfile(f, dtype=dtype, count=n_vert)
            xyz = np.stack([data[c] for c in ("x", "y", "z")], -1).astype(np.float32)
            rgb = np.stack([data[c] for c in ("red", "green", "blue")], -1).astype(
                np.uint8
            )
        else:
            raise ValueError(f"Unsupported ply format: {fmt}")
    return xyz, rgb


def save_ply_points(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write xyz + RGB as a binary_little_endian ply."""
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "<u1"),
            ("green", "<u1"),
            ("blue", "<u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def knn_indices_backend(points: Tensor, k: int, ref_max: int = 200_000) -> Tensor:
    """[N, k] nearest-neighbor indices (excluding self), open3d if available.

    The returned indices always address the ORIGINAL cloud, even when the torch
    fallback searches against a random reference subsample of at most
    ``ref_max`` points (chunked matmul topk) — an approximation for very large
    clouds, mirroring the online curvature estimation of GTLRStrategy. Self
    matches are excluded by identity, not by dropping the closest neighbor.
    """
    n = points.shape[0]
    if o3d is not None:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.numpy()))
        tree = o3d.geometry.KDTreeFlann(pcd)
        idx = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            _, nb, _ = tree.search_knn_vector_3d(pcd.points[i], k + 1)
            idx[i] = np.asarray(nb, dtype=np.int64)[1:]
        return torch.from_numpy(idx)
    ref_ids = torch.arange(n)
    if n > ref_max:
        ref_ids = torch.randperm(n)[:ref_max]
    from gsplat.strategy.gtlr import knn_indices

    local = knn_indices(
        points,
        points[ref_ids],
        k,
        query_ids=torch.arange(n),
        ref_ids=ref_ids,
    )
    return ref_ids[local]


def curvature_texture(
    points: Tensor, colors: Tensor, k: int = 64
) -> tuple[Tensor, Tensor]:
    """Per-point curvature kappa_i (Eq. 1-2) and texture tau_i (Eq. 3).

    Args:
        points: [N, 3] float. colors: [N, 3] float in [0, 1].

    Returns:
        (kappa [N], tau [N]), both raw (unnormalized).
    """
    idx = knn_indices_backend(points, k)
    nbrs = points[idx]  # [N, k, 3]
    centered = nbrs - nbrs.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / max(k - 1, 1)
    eig = torch.linalg.eigvalsh(cov)
    kappa = eig[..., 0] / (eig.sum(-1) + 1e-12)
    # Eq. 3: mean over the 3 channels of the neighborhood color variance.
    tau = colors[idx].var(dim=1, unbiased=False).mean(-1)
    return kappa, tau


def minmax_normalize(x: Tensor) -> Tensor:
    span = x.max() - x.min()
    if span < 1e-12:
        return torch.zeros_like(x)
    return (x - x.min()) / span


def sampling_probabilities(kappa: Tensor, tau: Tensor) -> Tensor:
    """Eq. 4 with alpha = beta = 0.5 on min-max normalized kappa and tau.

    Degenerate clouds (constant curvature AND constant texture, or non-finite
    values) carry no information; fall back to a uniform distribution.
    """
    p = 0.5 * minmax_normalize(kappa) + 0.5 * minmax_normalize(tau)
    total = p.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(p, 1.0 / p.numel())
    return p / total


def sample_indices(probs: np.ndarray, m: int, seed: int = 42) -> np.ndarray:
    """Draw m indices without replacement from the categorical distribution.

    numpy requires at least m non-zero probabilities for replace=False; when
    the support is too small, blend in a 1% uniform floor (a degenerate-case
    fallback, not part of the paper formulation).
    """
    n = probs.shape[0]
    m = min(m, n)
    if (probs > 0).sum() < m:
        probs = 0.99 * probs / probs.sum() + 0.01 / n
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=m, replace=False, p=probs / probs.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_ply", required=True, help="Registered LiDAR ply with RGB")
    parser.add_argument("--output_ply", required=True, help="Sampled init ply")
    parser.add_argument("--num_samples", type=int, default=3_000_000)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    xyz, rgb = load_ply_points(args.input_ply)
    points = torch.from_numpy(xyz)
    colors = torch.from_numpy(rgb).float() / 255.0
    kappa, tau = curvature_texture(points, colors, k=args.knn)
    probs = sampling_probabilities(kappa, tau).numpy()
    idx = sample_indices(probs, args.num_samples, seed=args.seed)
    save_ply_points(args.output_ply, xyz[idx], rgb[idx])
    # Coordinate-frame sidecar: the trainer uses it to decide whether the
    # parser normalization still has to be applied (never twice).
    meta = {
        "coordinate_frame": "raw",
        "source_ply": os.path.basename(args.input_ply),
        "num_samples": int(len(idx)),
        "knn": args.knn,
        "seed": args.seed,
    }
    with open(args.output_ply + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Sampled {len(idx)} / {len(xyz)} points -> {args.output_ply}")


if __name__ == "__main__":
    main()
