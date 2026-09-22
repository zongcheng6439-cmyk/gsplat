# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GTLR-GS curvature-adaptive splitting strategy (arXiv:2603.23192).

Implements the curvature gate of GTLR-GS on top of
:class:`gsplat.strategy.DefaultStrategy`: a split candidate additionally has to
exhibit a high *online* local curvature (Eq. 1-2 of the paper, recomputed on the
current Gaussian centers),

    kappa_i = lambda_1 / (lambda_1 + lambda_2 + lambda_3 + eps),

where lambda_1 <= lambda_2 <= lambda_3 are the eigenvalues of the kNN covariance
of the neighborhood. The gate follows the linear schedule (Eq. 5-6)

    theta(t) = theta_start + (theta_end - theta_start) * t / T,

with theta_start = 0.1, theta_end = 0.3 and T the total training iterations.
Note the raw range of kappa is [0, 1/3]; it is intentionally *not* renormalized.

Performance note (approximation): an exact all-pairs kNN on ~3M Gaussians at
every refine step is infeasible, so curvature is only evaluated for the actual
split candidates (`is_grad_high & is_large`), and the neighbor reference set is
a uniform random subsample of the Gaussian centers capped at
``curvature_ref_max`` (default 200k). Nearest neighbors are found with chunked
matmul + topk on GPU. This trades a small estimation bias for an O(|cand| x
|ref|) cost instead of O(N^2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from torch import Tensor

from .default import DefaultStrategy
from .ops import duplicate, split

_COV_EPS = 1e-12


def curvature_threshold(
    step: int, total_iters: int, start: float = 0.1, end: float = 0.3
) -> float:
    """Linear threshold schedule theta(t) of Eq. 5-6, clamped to [start, end]."""
    t = min(max(step, 0), total_iters) / max(total_iters, 1)
    return start + (end - start) * t


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


def knn_curvature(
    query: Tensor,
    ref: Tensor,
    k: int,
    chunk_size: int = 4096,
    query_ids: Tensor | None = None,
    ref_ids: Tensor | None = None,
) -> Tensor:
    """Surface-variation curvature kappa (Eq. 1-2) of ``query`` against ``ref``.

    kappa = lambda_1 / (lambda_1 + lambda_2 + lambda_3 + eps) in [0, 1/3],
    from the eigenvalues of the covariance of the k nearest neighbors.

    Pass ``query_ids``/``ref_ids`` (identity ids in a common point set) so an
    exact self match is excluded from the neighborhood instead of blindly
    dropping the closest neighbor.
    """
    idx = knn_indices(
        query, ref, k, chunk_size=chunk_size, query_ids=query_ids, ref_ids=ref_ids
    )
    nbrs = ref[idx]  # [Q, k, 3]
    centered = nbrs - nbrs.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / max(k - 1, 1)  # [Q, 3, 3]
    eig = torch.linalg.eigvalsh(cov)  # ascending
    return eig[..., 0] / (eig.sum(-1) + _COV_EPS)


@dataclass
class GTLRStrategy(DefaultStrategy):
    """DefaultStrategy with GTLR-GS curvature-adaptive splitting (arXiv:2603.23192).

    Everything follows :class:`DefaultStrategy` (duplication, pruning, opacity
    reset are unchanged) except that a split candidate is additionally gated by
    its online curvature: a Gaussian is split only if its kNN-covariance
    curvature kappa_i (Eq. 1-2, estimated on the current Gaussian centers)
    exceeds the scheduled threshold theta(t) (Eq. 5-6).

    Approximation: curvature is computed only for split candidates, against a
    random subsample of the Gaussian centers capped at ``curvature_ref_max``,
    via chunked matmul topk (see the module docstring).

    Args:
        curvature_start (float): theta_start of the split-gate schedule. Default 0.1.
        curvature_end (float): theta_end of the split-gate schedule. Default 0.3.
        curvature_knn (int): k of the kNN neighborhood for online curvature. Default 16.
        curvature_ref_max (int): Max size of the random reference subsample used
          for neighbor search. Default 200_000.
        total_iters (int): T of the schedule; total training iterations. Default 30_000.
    """

    curvature_start: float = 0.1
    curvature_end: float = 0.3
    curvature_knn: int = 16
    curvature_ref_max: int = 200_000
    total_iters: int = 30_000

    def split_gate(self, means: Tensor, candidates: Tensor, step: int) -> Tensor:
        """Curvature gate for split candidates; returns a bool mask like ``candidates``.

        kappa is evaluated only at candidate positions; the neighbor reference
        set is a random subsample of ``means`` capped at ``curvature_ref_max``.
        """
        device = means.device
        cand_idx = torch.where(candidates)[0]
        if len(cand_idx) == 0:
            return candidates
        ref_idx = torch.randperm(means.shape[0], device=device)[
            : self.curvature_ref_max
        ]
        ref = means[ref_idx]
        kappa = knn_curvature(
            means[cand_idx],
            ref,
            self.curvature_knn,
            query_ids=cand_idx,
            ref_ids=ref_idx,
        )
        theta = curvature_threshold(
            step, self.total_iters, self.curvature_start, self.curvature_end
        )
        gate = torch.zeros_like(candidates)
        gate[cand_idx] = kappa > theta
        return gate

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        scene=None,
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        # GTLR-GS curvature gate (Eq. 5-6): split only where kappa_i > theta(t).
        is_split = is_split & self.split_gate(
            params["means"].detach(), is_split, step
        )
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_dupli,
                scene=scene,
            )

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
                scene=scene,
            )
        return n_dupli, n_split
