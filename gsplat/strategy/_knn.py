# SPDX-License-Identifier: Apache-2.0
"""Standalone chunked kNN (pure torch, no gsplat package imports).

Kept in its own leaf module so offline tools (e.g. ``examples/gtlr`` data
preprocessing) can load it by file path WITHOUT importing the ``gsplat``
package — importing the package triggers the CUDA JIT build, which the
preprocessing does not need.
"""

from __future__ import annotations

import torch
from torch import Tensor


def knn_indices(
    query: Tensor,
    ref: Tensor,
    k: int,
    chunk_size: int = 4096,
    query_ids: Tensor | None = None,
    ref_ids: Tensor | None = None,
) -> Tensor:
    """Indices of the k nearest neighbors in ``ref`` for each row of ``query``.

    Chunked |q|^2 - 2 q.r + |r|^2 matmul with topk; memory stays O(chunk x R).
    The returned indices are always row indices into ``ref``; callers needing
    indices into a parent cloud must map them back through their ``ref_ids``.

    Args:
        query: [Q, 3] points. ref: [R, 3] reference points. k: neighbors.
        query_ids/ref_ids: optional identity ids of the query/ref rows within a
            common point set. When both are given, an exact self match (same
            id) is excluded per row by fetching k+1 neighbors and dropping the
            match; without ids no self exclusion is performed.

    Returns:
        [Q, k] int64 indices into ``ref``, sorted by ascending distance.
    """
    k = min(k, ref.shape[0])
    exclude_self = query_ids is not None and ref_ids is not None
    kk = min(k + 1, ref.shape[0]) if exclude_self else k
    ref_norm = (ref * ref).sum(-1)
    out = []
    for i, q in enumerate(query.split(chunk_size)):
        d2 = (q * q).sum(-1, keepdim=True) - 2.0 * q @ ref.T + ref_norm
        idx = d2.topk(kk, dim=-1, largest=False).indices
        if exclude_self and kk > k:
            qids = query_ids[i * chunk_size : i * chunk_size + len(q)]
            match = ref_ids[idx] == qids[:, None]  # [q, kk], at most one True
            # stable argsort pushes the self match (1) to the end
            order = torch.argsort(match.to(torch.int8), dim=1, stable=True)
            idx = torch.gather(idx, 1, order)[:, :k]
        out.append(idx)
    return torch.cat(out, dim=0)
