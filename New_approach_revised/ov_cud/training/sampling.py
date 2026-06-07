"""Pair sampling for the relation head (design.md sec 10.5).

Stratifies candidate pairs into same-instance, same-class/diff-instance,
different-class, part-whole, hard-negative (appearance-similar / different
class), and background pairs, then samples up to ``max_pairs`` with rough class
balance. Returns a list of (i, j) ordered pairs.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..matrix.pairwise_features import PairwiseContext
from .matching import MatchResult

PairList = List[Tuple[int, int]]


def _categorize(match: MatchResult, ctx: PairwiseContext) -> dict:
    n = match.is_valid.shape[0]
    buckets = {
        "same_instance": [], "same_class_diff_instance": [], "different_class": [],
        "part_whole": [], "hard_negative": [], "background": [],
    }
    cls = match.matched_class
    inst = match.matched_instance_id
    valid = match.is_valid
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if not (valid[i] and valid[j]):
                buckets["background"].append((i, j))
                continue
            same_inst = inst[i] >= 0 and inst[i] == inst[j]
            same_cls = cls[i] != "" and cls[i] == cls[j]
            if same_inst:
                # part-whole if one clearly contains the other and coverage differs
                if ctx.containment[i, j] > 0.5 and match.coverage[j] > match.coverage[i]:
                    buckets["part_whole"].append((i, j))
                else:
                    buckets["same_instance"].append((i, j))
            elif same_cls:
                buckets["same_class_diff_instance"].append((i, j))
            else:
                if ctx.cos_z[i, j] > 0.5:
                    buckets["hard_negative"].append((i, j))
                else:
                    buckets["different_class"].append((i, j))
    return buckets


def sample_pairs(
    match: MatchResult, ctx: PairwiseContext, max_pairs: int = 512, seed: int = 0
) -> PairList:
    if ctx.n <= 1:
        return []
    buckets = _categorize(match, ctx)
    rng = np.random.default_rng(seed)
    active = {k: v for k, v in buckets.items() if v}
    if not active:
        return []

    per = max(1, max_pairs // len(active))
    out: PairList = []
    for _name, pairs in active.items():
        if len(pairs) <= per:
            out.extend(pairs)
        else:
            idx = rng.choice(len(pairs), size=per, replace=False)
            out.extend(pairs[k] for k in idx)
    return out
