"""Label generation from matching results (design.md sec 10.3, 10.4).

Category targets (Stage 1) and relation pair targets (Stage 2) are derived
entirely from {matched_class, matched_instance_id, purity, coverage} plus the
pairwise containment from the PairwiseContext.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..vocabulary import VocabularyBank
from .matching import MatchResult

IGNORE_INDEX = -1


def category_targets(match: MatchResult, vocab: VocabularyBank) -> Tuple[np.ndarray, np.ndarray]:
    """Return (class_idx [N] with -1 = ignore, weight [N]).

    Candidates whose matched class is not in the vocabulary, or that are
    background/invalid, are ignored (weight 0, index -1).
    """
    names = set(vocab.class_names)
    n = len(match.matched_class)
    idx = np.full(n, IGNORE_INDEX, dtype=np.int64)
    weight = np.zeros(n, dtype=np.float32)
    for i in range(n):
        cls = match.matched_class[i]
        if match.is_valid[i] and cls in names:
            idx[i] = vocab.index_of(cls)
            weight[i] = float(match.weight[i])
    return idx, weight


def relation_pair_targets(
    match: MatchResult,
    containment: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
) -> dict:
    """Vectorized y_sem / y_inst / y_part and pair weights for given pairs."""
    if len(pairs) == 0:
        z = np.zeros(0, dtype=np.float32)
        return {"y_sem": z, "y_inst": z, "y_part": z, "weight": z}

    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)

    valid = match.is_valid
    both_valid = valid[ii] & valid[jj]

    cls = np.array(match.matched_class, dtype=object)
    same_class = (cls[ii] == cls[jj]) & (cls[ii] != "")

    inst = match.matched_instance_id
    same_inst = (inst[ii] == inst[jj]) & (inst[ii] >= 0)

    y_sem = (both_valid & same_class).astype(np.float32)
    y_inst = (both_valid & same_inst).astype(np.float32)

    # part soft target: same_inst * containment_i_in_j * completeness_gap
    cont_ij = containment[ii, jj]
    completeness_gap = np.clip(match.coverage[jj] - match.coverage[ii], 0.0, 1.0)
    y_part = (same_inst.astype(np.float32) * cont_ij * completeness_gap).astype(np.float32)

    weight = (match.weight[ii] * match.weight[jj]).astype(np.float32)
    return {"y_sem": y_sem, "y_inst": y_inst, "y_part": y_part, "weight": weight}
