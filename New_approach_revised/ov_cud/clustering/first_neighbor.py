"""Stage 6: category-aware first-neighbor clustering (design.md sec 11.2).

Steps:
  1. (filtering already done upstream)
  2. bucket candidates by top-1 class (bucket_top_k allows multi-bucket)
  3. within each bucket run first-neighbor clustering on the A_group submatrix
  4. if max_j A_group[i, j] < tau_affinity -> singleton (unknown/noise handled later)
  5. connected components -> coarse semantic groups

Returns a list of groups, each a sorted list of *global* candidate indices,
covering every input candidate exactly once.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import ClusterConfig
from .connected_components import connected_components


def first_neighbor_graph(a_group: np.ndarray, tau_affinity: float) -> np.ndarray:
    """Each node links to its strongest neighbor if that edge >= tau."""
    n = a_group.shape[0]
    graph = np.zeros((n, n), dtype=bool)
    for i in range(n):
        scores = a_group[i].copy()
        scores[i] = -np.inf
        if n <= 1:
            continue
        j = int(np.argmax(scores))
        if scores[j] >= tau_affinity:
            graph[i, j] = True
            graph[j, i] = True
    return graph


def first_neighbor_clustering(a_group: np.ndarray, tau_affinity: float) -> List[List[int]]:
    if a_group.shape[0] == 0:
        return []
    graph = first_neighbor_graph(a_group, tau_affinity)
    return connected_components(graph)


def _buckets(category_probs: np.ndarray, top_k: int, valid: np.ndarray) -> dict[int, List[int]]:
    buckets: dict[int, List[int]] = {}
    for i in range(category_probs.shape[0]):
        if not valid[i]:
            continue
        order = np.argsort(category_probs[i])[::-1][:max(1, top_k)]
        for c in order:
            buckets.setdefault(int(c), []).append(i)
    return buckets


def category_aware_clustering(
    category_probs: np.ndarray,
    a_group: np.ndarray,
    config: ClusterConfig,
    valid_mask: Optional[np.ndarray] = None,
) -> List[List[int]]:
    n = category_probs.shape[0]
    if n == 0:
        return []
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=bool)

    # candidates whose top score is too low are isolated as singletons.
    top_score = category_probs.max(axis=1)
    valid = valid_mask & (top_score >= config.min_top_score)

    groups: List[List[int]] = []
    assigned = set()
    buckets = _buckets(category_probs, config.bucket_top_k, valid)
    for _cls, members in sorted(buckets.items()):
        members = [m for m in members if m not in assigned]
        if not members:
            continue
        sub = a_group[np.ix_(members, members)]
        for comp in first_neighbor_clustering(sub, config.tau_affinity):
            g = sorted(members[k] for k in comp)
            groups.append(g)
            assigned.update(g)

    # any candidate not assigned (invalid / low score) -> its own singleton.
    for i in range(n):
        if i not in assigned:
            groups.append([i])
            assigned.add(i)
    return groups
