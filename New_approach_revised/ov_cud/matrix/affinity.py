"""Stage 6 affinity (design.md sec 11.1).

    category_compatibility_ij = p_i . p_j
    A_group_ij = category_compatibility_ij * semantic_relation_ij

A_sem from the relation head is already a probability in [0, 1], so it is used
directly as ``semantic_relation`` (set ``apply_sigmoid=True`` only if a head
emits raw logits). A_inst / A_part are NOT used for grouping -- only later for
dedup / part filtering / representative selection.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_group_affinity(
    category_probs: np.ndarray, a_sem: np.ndarray, apply_sigmoid: bool = False
) -> np.ndarray:
    if category_probs.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    compat = category_probs @ category_probs.T
    sem = _sigmoid(a_sem) if apply_sigmoid else a_sem
    a_group = compat * sem
    np.fill_diagonal(a_group, 0.0)
    return a_group.astype(np.float32)
