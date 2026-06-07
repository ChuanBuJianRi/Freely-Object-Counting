"""Stage 5: pairwise relation head (design.md sec 9).

Outputs three N x N matrices for a ``PairwiseContext``:
    A_sem[i,j]  = P(same semantic class)   (symmetric)
    A_inst[i,j] = P(same real instance)    (symmetric)
    A_part[i,j] = P(c_i is part of c_j)    (directional)

Two implementations share one interface so the trained head drops in for the
heuristic stub without touching the pipeline:
  - HeuristicRelationHead: geometric/appearance proxies (no training). Default
    when no trained checkpoint is available.
  - LearnedRelationHead: an MLP over phi_ij (trained by training/train_relation).
"""

from __future__ import annotations

from typing import Dict, Protocol

import numpy as np

from ..matrix.pairwise_features import (
    PairwiseContext,
    all_ordered_pairs,
    build_phi_pairs,
    phi_dim,
)


class RelationHead(Protocol):
    def __call__(self, ctx: PairwiseContext) -> Dict[str, np.ndarray]: ...


def _zero_diag(m: np.ndarray) -> np.ndarray:
    m = m.copy()
    np.fill_diagonal(m, 0.0)
    return m


class HeuristicRelationHead:
    """Training-free geometric proxy relation head (stub).

    A_sem  = (cos(z_i, z_j) + 1) / 2   appearance similarity in [0, 1]
    A_inst = IoU(mask_i, mask_j)        duplicate masks ~ same instance
    A_part = containment_i_in_j         i inside j ~ i is part of j (directional)
    """

    def __call__(self, ctx: PairwiseContext) -> Dict[str, np.ndarray]:
        if ctx.n == 0:
            z = np.zeros((0, 0), dtype=np.float32)
            return {"A_sem": z, "A_inst": z, "A_part": z}
        a_sem = np.clip((ctx.cos_z + 1.0) / 2.0, 0.0, 1.0)
        a_inst = np.clip(ctx.iou_mask, 0.0, 1.0)
        a_part = np.clip(ctx.containment, 0.0, 1.0)
        return {
            "A_sem": _zero_diag(a_sem).astype(np.float32),
            "A_inst": _zero_diag(a_inst).astype(np.float32),
            "A_part": _zero_diag(a_part).astype(np.float32),
        }


class LearnedRelationHead:
    """MLP over phi_ij -> (sem, inst, part) logits. Same interface as the stub.

    sem/inst are symmetrized (average of i->j and j->i); part stays directional.
    Runs the full N^2 pairs (no pruning).
    """

    def __init__(self, module, z_dim: int, device: str = "cpu", batch_size: int = 8192):
        self.module = module
        self.z_dim = z_dim
        self.device = device
        self.batch_size = batch_size

    def __call__(self, ctx: PairwiseContext) -> Dict[str, np.ndarray]:
        import torch

        n = ctx.n
        a_sem = np.zeros((n, n), dtype=np.float32)
        a_inst = np.zeros((n, n), dtype=np.float32)
        a_part = np.zeros((n, n), dtype=np.float32)
        if n == 0:
            return {"A_sem": a_sem, "A_inst": a_inst, "A_part": a_part}

        pairs = all_ordered_pairs(n, include_self=False)
        self.module.eval()
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start:start + self.batch_size]
                phi = build_phi_pairs(ctx, chunk)
                x = torch.from_numpy(phi).to(self.device)
                probs = torch.sigmoid(self.module(x)).cpu().numpy()  # [P, 3]
                for k, (i, j) in enumerate(chunk):
                    a_sem[i, j] = probs[k, 0]
                    a_inst[i, j] = probs[k, 1]
                    a_part[i, j] = probs[k, 2]

        # symmetrize sem/inst; part stays directional.
        a_sem = (a_sem + a_sem.T) / 2.0
        a_inst = (a_inst + a_inst.T) / 2.0
        return {
            "A_sem": _zero_diag(a_sem),
            "A_inst": _zero_diag(a_inst),
            "A_part": _zero_diag(a_part),
        }


def build_relation_mlp(z_dim: int, hidden: int = 256):
    """Small MLP: phi_dim -> hidden -> hidden -> 3 logits (sem, inst, part)."""
    import torch.nn as nn

    f = phi_dim(z_dim)
    return nn.Sequential(
        nn.Linear(f, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 3),
    )
