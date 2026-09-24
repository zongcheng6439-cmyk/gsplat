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

from which the target M points are drawn and written back out as a
gsplat-compatible xyz+RGB ply. The paper's allocation is down-sampling only
(it assumes M << N, tens of millions of LiDAR points): if num_samples > N the
script exits with an error, unless ``--allow_upsample`` is given — an
extension that keeps all N points and draws the remainder WITH replacement
(high-score points replicate); each replicated point is placed by linear
interpolation between its parent and one of the parent's nearest neighbors
(conventional on-surface up-sampling), with interpolated color. Optionally, a ``base_ratio`` fraction of M is drawn
uniformly for coverage of flat low-score regions (0 = pure paper sampling).

kNN backend: open3d when installed, otherwise chunked torch matmul topk
against a random reference subsample (same approximation as the strategy).

Usage:
    python -m gtlr.sample_points --input_ply fused.ply --output_ply init_3m.ply \
        [--num_samples 3000000] [--allow_upsample]
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

try:  # plain script from the gtlr directory
    from knn import knn_indices
except ImportError:  # run as `python -m gtlr.sample_points` from examples/
    from gtlr.knn import knn_indices

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


def knn_indices_backend(
    points: Tensor, k: int, ref_max: int = 200_000, chunk_size: int = 4096
) -> Tensor:
    """[N, k] nearest-neighbor indices (excluding self), open3d if available.

    The returned indices always address the ORIGINAL cloud, even when the torch
    fallback searches against a random reference subsample of at most
    ``ref_max`` points (chunked matmul topk) — an approximation for very large
    clouds, mirroring the online curvature estimation of GTLRStrategy. Self
    matches are excluded by identity, not by dropping the closest neighbor.
    ``chunk_size`` rows are scored at a time; the transient distance matrix is
    ``chunk_size x ref`` float32 (4096 x 200k ~= 3.3 GB).
    """
    n = points.shape[0]
    if o3d is not None:
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(points.cpu().numpy())
        )
        tree = o3d.geometry.KDTreeFlann(pcd)
        idx = np.empty((n, k), dtype=np.int64)
        for i in range(n):
            _, nb, _ = tree.search_knn_vector_3d(pcd.points[i], k + 1)
            idx[i] = np.asarray(nb, dtype=np.int64)[1:]
        return torch.from_numpy(idx).to(points.device)
    ref_ids = torch.arange(n, device=points.device)
    if n > ref_max:
        ref_ids = torch.randperm(n, device=points.device)[:ref_max]

    local = knn_indices(
        points,
        points[ref_ids],
        k,
        chunk_size=chunk_size,
        query_ids=torch.arange(n, device=points.device),
        ref_ids=ref_ids,
    )
    return ref_ids[local]


def curvature_texture(
    points: Tensor,
    colors: Tensor,
    k: int = 64,
    ref_max: int = 200_000,
    chunk_size: int = 4096,
    return_idx: bool = False,
):
    """Per-point curvature kappa_i (Eq. 1-2) and texture tau_i (Eq. 3).

    Args:
        points: [N, 3] float. colors: [N, 3] float in [0, 1].
        return_idx: also return the [N, k] neighbor indices (the caller can
            reuse them for up-sampling instead of a second kNN).

    Returns:
        (kappa [N], tau [N]), both raw (unnormalized); with return_idx,
        (kappa, tau, idx).
    """
    idx = knn_indices_backend(points, k, ref_max=ref_max, chunk_size=chunk_size)
    # Chunked over query points: the [chunk, k, 3] neighbor gather is the
    # memory peak next to the kNN distance matrix, so keep it bounded.
    # The eigendecomposition is further split at 8192: cusolver's batched
    # syev fails with CUSOLVER_STATUS_INVALID_VALUE beyond ~16k batches
    # (observed on cu128 / sm_75).
    n = points.shape[0]
    kappa = torch.empty(n, device=points.device)
    tau = torch.empty(n, device=points.device)
    cov_chunk = 262_144
    for s in range(0, n, cov_chunk):
        sub = idx[s : s + cov_chunk]
        nbrs = points[sub]  # [c, k, 3]
        centered = nbrs - nbrs.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / max(k - 1, 1)
        eig = torch.cat([torch.linalg.eigvalsh(c) for c in cov.split(8192)])
        kappa[s : s + cov_chunk] = eig[..., 0] / (eig.sum(-1) + 1e-12)
        # Eq. 3: mean over the 3 channels of the neighborhood color variance.
        tau[s : s + cov_chunk] = colors[sub].var(dim=1, unbiased=False).mean(-1)
        del nbrs, centered, cov, eig
    if return_idx:
        return kappa, tau, idx
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
    """Pure score-proportional draw (base_ratio = 0), kept for compatibility.

    m <= n: m indices without replacement. m > n: all n indices (in order),
    then m - n extra draws WITH replacement — high-score points replicate.
    """
    base, real, dup = hybrid_sample_indices(probs, m, base_ratio=0.0, seed=seed)
    return np.concatenate([base, real, dup])


def hybrid_sample_indices(
    probs: np.ndarray, m: int, base_ratio: float = 0.0, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-stage sampling: uniform coverage base + score-weighted detail.

    Args:
        probs: [N] sampling probability (Eq. 4). m: target count.
        base_ratio: fraction of m drawn UNIFORMLY (coverage of low-scoring
            flat regions); the rest follows ``probs`` (detail in
            high-curvature / high-texture regions). 0 reproduces the pure
            score-proportional sampling of the paper.

    Returns:
        (base_idx, real_idx, dup_idx): uniform base (without replacement),
        score-weighted real points (without replacement from the remainder),
        and score-weighted duplicates (WITH replacement, only once every real
        point is already used and m is still not reached — the caller places
        them by neighbor interpolation, see ``upsample_interpolate``).
    """
    n = probs.shape[0]
    p = probs / probs.sum()
    rng = np.random.default_rng(seed)

    base_n = min(n, int(m * base_ratio))
    if base_n:
        base = rng.choice(n, size=base_n, replace=False)
    else:
        base = np.empty(0, dtype=np.int64)

    rest = np.setdiff1d(np.arange(n), base)  # sorted
    detail_n = m - base_n
    real_n = min(detail_n, len(rest))
    if real_n == len(rest):
        real = rest  # every remaining real point is used
    elif real_n > 0:
        pr = p[rest]
        if (pr > 0).sum() < real_n:
            # degenerate support: blend in a 1% uniform floor
            pr = 0.99 * pr / pr.sum() + 0.01 / len(rest)
        real = rest[rng.choice(len(rest), size=real_n, replace=False, p=pr / pr.sum())]
    else:
        real = np.empty(0, dtype=np.int64)

    dup_n = detail_n - real_n
    if dup_n > 0:
        dup = rng.choice(n, size=dup_n, replace=True, p=p)
    else:
        dup = np.empty(0, dtype=np.int64)
    return base, real, dup


def upsample_interpolate(
    points: Tensor,
    colors: Tensor,
    nn_idx: Tensor,
    extra_idx: np.ndarray,
    neighbor_pool: int = 8,
    seed: int = 42,
) -> tuple[Tensor, Tensor]:
    """Conventional point-cloud up-sampling by neighbor-edge interpolation.

    Each new point is placed at a uniformly random position on the segment
    between a parent (drawn by score) and one of its ``neighbor_pool``
    nearest neighbors: ``p_new = (1 - t) * p_parent + t * p_nbr``,
    t ~ U(0, 1). Added points stay on the local surface and inherit the
    interpolated color. ``nn_idx`` reuses the curvature kNN result
    ([N, k], self excluded) so no second kNN pass is needed.
    """
    pool = nn_idx[:, : min(neighbor_pool, nn_idx.shape[1])]  # [N, P]
    parent = torch.from_numpy(extra_idx).long().to(points.device)
    rng = np.random.default_rng(seed + 1)
    choice = torch.from_numpy(
        rng.integers(0, pool.shape[1], len(extra_idx)).astype(np.int64)
    ).to(points.device)
    nbr = pool[parent, choice]  # [E]
    t = torch.from_numpy(rng.random(len(extra_idx)).astype(np.float32)).to(
        points.device
    )[:, None]
    new_xyz = points[parent] * (1.0 - t) + points[nbr] * t
    new_rgb = colors[parent] * (1.0 - t) + colors[nbr] * t
    return new_xyz, new_rgb


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Only the four options above are needed; all other knobs have "
        "sensible defaults (knn=64, seed=42, ref_max=50k, chunk_size=1024, "
        "device=auto, upsample_neighbors=8, base_ratio=0).",
    )
    parser.add_argument("--input_ply", required=True, help="Registered LiDAR ply with RGB")
    parser.add_argument("--output_ply", required=True, help="Sampled init ply")
    parser.add_argument("--num_samples", type=int, default=3_000_000)
    parser.add_argument(
        "--allow_upsample",
        action="store_true",
        help="Required when num_samples > N: keep all points and add new ones "
        "by neighbor interpolation, allocated by score. The paper's "
        "allocation (Eq. 4) is down-sampling only and assumes M << N.",
    )
    # Advanced knobs: hidden from --help, defaults are right for this pipeline.
    # ref_max/chunk_size default to shared-GPU-safe values: the transient
    # distance matrix is chunk_size x ref_max x 4B ~= 200 MiB, so sampling
    # survives alongside other GPU jobs.
    parser.add_argument("--knn", type=int, default=64, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--ref_max", type=int, default=50_000, help=argparse.SUPPRESS)
    parser.add_argument("--chunk_size", type=int, default=1024, help=argparse.SUPPRESS)
    parser.add_argument("--device", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--upsample_neighbors", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--base_ratio", type=float, default=0.0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    xyz, rgb = load_ply_points(args.input_ply)
    n = len(xyz)
    if args.num_samples > n and not args.allow_upsample:
        parser.exit(
            1,
            f"Error: --num_samples {args.num_samples} exceeds the input cloud "
            f"size N={n}.\nGTLR-GS's geometry-texture allocation (Eq. 4) is a "
            "down-sampling scheme and assumes M << N (tens of millions of "
            "LiDAR points).\nOptions:\n"
            f"  - lower --num_samples below {n},\n"
            "  - use a denser input point cloud,\n"
            "  - or pass --allow_upsample to keep all N points and add new "
            "ones by neighbor interpolation (extension beyond the paper).\n",
    )
    device = args.device
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("cuda not available, falling back to cpu")
        device = "cpu"
    print(f"device: {device}")
    points = torch.from_numpy(xyz).to(device)
    colors = torch.from_numpy(rgb).float().to(device) / 255.0
    kappa, tau, nn_idx = curvature_texture(
        points,
        colors,
        k=args.knn,
        ref_max=args.ref_max,
        chunk_size=args.chunk_size,
        return_idx=True,
    )
    probs = sampling_probabilities(kappa, tau).cpu().numpy()
    idx_base, idx_real, idx_dup = hybrid_sample_indices(
        probs, args.num_samples, base_ratio=args.base_ratio, seed=args.seed
    )
    parts_xyz = [xyz[idx_base], xyz[idx_real]]
    parts_rgb = [rgb[idx_base], rgb[idx_real]]
    if len(idx_dup):
        # Conventional point-cloud up-sampling: each new point is a linear
        # interpolation between a (score-sampled) parent and one of its
        # nearest neighbors, so added points lie on the local surface.
        new_xyz, new_rgb = upsample_interpolate(
            points,
            colors,
            nn_idx,
            idx_dup,
            neighbor_pool=args.upsample_neighbors,
            seed=args.seed,
        )
        parts_xyz.append(new_xyz.cpu().numpy())
        parts_rgb.append((new_rgb.cpu().numpy() * 255.0).round().astype(np.uint8))
        print(
            f"Up-sampled {len(idx_dup)} points (interpolation over "
            f"{args.upsample_neighbors} nearest neighbors)"
        )
    out_xyz = np.concatenate(parts_xyz)
    out_rgb = np.concatenate(parts_rgb)
    save_ply_points(args.output_ply, out_xyz, out_rgb)
    # Coordinate-frame sidecar: the trainer uses it to decide whether the
    # parser normalization still has to be applied (never twice).
    meta = {
        "coordinate_frame": "raw",
        "source_ply": os.path.basename(args.input_ply),
        "num_samples": int(len(out_xyz)),
        "num_base": int(len(idx_base)),
        "num_real": int(len(idx_real)),
        "num_upsampled": int(len(idx_dup)),
        "base_ratio": args.base_ratio,
        "upsample_neighbors": args.upsample_neighbors if len(idx_dup) else 0,
        "knn": args.knn,
        "ref_max": args.ref_max,
        "chunk_size": args.chunk_size,
        "seed": args.seed,
    }
    with open(args.output_ply + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"Sampled {len(out_xyz)} / {len(xyz)} points "
        f"(base {len(idx_base)}, real {len(idx_real)}, dup {len(idx_dup)}) "
        f"-> {args.output_ply}"
    )


if __name__ == "__main__":
    main()
