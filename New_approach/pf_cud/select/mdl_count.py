"""Adaptive, parameter-free count selection via Minimum Description Length.

Motivation
----------
The default pipeline ends with ``rank_groups`` + "take rank-1 group size",
which is itself a hidden policy (pick one group) and is biased toward small,
ultra-consistent groups. Swapping in another fixed rule (largest / total / ...)
is just another policy.

This module replaces the *decision* with model selection: we never tell the
method "take the largest group" or "sum everything". Instead we ask a single
question whose answer is read off the data distribution:

    Out of all the ways to partition the candidates into K repeated-unit
    types, which K (and which partition) yields the shortest total description
    length L(K)?

We obtain the family of partitions for free from the fused-distance MST: cutting
the k longest MST edges yields k+1 connected components. Sweeping k = 0..n-1
traces a full L(k) curve; argmin selects the partition. No threshold (unlike a
single Otsu cut), no tuned k, no per-dataset knob -- the curve's shape is set by
the image's own feature distribution, so K* adapts per image.

Counting then also comes from MDL, not a policy: among the units of the optimal
partition, the count is the size of the unit type with the largest *compression
gain* -- the bits saved by encoding its members as "one prototype + N near-equal
instances" versus encoding them independently. The most compressible repeated
structure is, by definition, the dominant countable unit.
"""

from typing import List, Tuple

import numpy as np
from scipy.sparse.csgraph import connected_components

from pf_cud.data import Candidate, CountGroup
from pf_cud.features.fusion import fused_distance
from pf_cud.graph.mst import build_mst
from pf_cud.mdl.score import common_feature_keys, group_stats, mdl_from_stats


def _mst_edges_sorted(mst: np.ndarray) -> List[Tuple[float, int, int]]:
    """Return MST edges (weight, i, j) sorted by ascending weight."""
    n = mst.shape[0]
    edges = []
    for i in range(n):
        row = mst[i]
        for j in range(i + 1, n):
            w = row[j]
            if w > 0:
                edges.append((float(w), i, j))
    edges.sort(key=lambda e: e[0])
    return edges


def _partition_mdl(
    candidates: List[Candidate],
    labels: np.ndarray,
    keys: List[str],
    num_c: int,
) -> float:
    """Total description length of a labelling (sum of per-component MDL)."""
    total = 0.0
    for c in np.unique(labels):
        idx = np.where(labels == c)[0].tolist()
        total += mdl_from_stats(group_stats(candidates, idx, keys), num_c, keys)
    return total


def _unit_compression_gain(
    candidates: List[Candidate], indices: List[int], keys: List[str], num_c: int
) -> float:
    """Bits saved by coding members jointly (1 prototype + residuals) vs alone.

    gain = sum_i MDL({i}) - MDL(group). Large gain == many near-identical
    members == a strongly countable repeated unit. Singletons gain 0.
    """
    if len(indices) <= 1:
        return 0.0
    joint = mdl_from_stats(group_stats(candidates, indices, keys), num_c, keys)
    alone = sum(
        mdl_from_stats(group_stats(candidates, [i], keys), num_c, keys)
        for i in indices
    )
    return alone - joint


def mdl_select_count(candidates: List[Candidate]) -> dict:
    """Pick the count by MDL model selection over MST-induced partitions.

    Returns a dict with the chosen count, the optimal number of units K*, the
    full L(k) sweep, and the dominant unit's member indices -- all derived from
    the data, with no tunable parameter or selection policy.
    """
    n = len(candidates)
    if n == 0:
        return {"count": 0, "k_star": 0, "indices": [], "curve": []}
    if n == 1:
        return {"count": 1, "k_star": 1, "indices": [0], "curve": []}

    keys = common_feature_keys(candidates)
    d = fused_distance(candidates)
    mst = build_mst(d)
    edges = _mst_edges_sorted(mst)

    # Adjacency starts as the full MST; we remove the longest edges one at a
    # time (from the end of the sorted list), which monotonically increases the
    # number of connected components from 1 (k=0 cuts) to n (n-1 cuts).
    adj = np.zeros((n, n), dtype=np.int8)
    for _, i, j in edges:
        adj[i, j] = adj[j, i] = 1

    # Evaluate L(k) for every cut count k = 0..len(edges).
    curve = []
    best = None  # (mdl, k, labels)
    removable = list(reversed(edges))  # longest first
    for k in range(len(edges) + 1):
        _, labels = connected_components(adj, directed=False)
        mdl = _partition_mdl(candidates, labels, keys, n)
        curve.append((k, int(labels.max() + 1), float(mdl)))
        if best is None or mdl < best[0]:
            best = (mdl, k, labels.copy())
        if k < len(removable):
            _, i, j = removable[k]
            adj[i, j] = adj[j, i] = 0

    _, k_star_cuts, labels = best
    k_units = int(labels.max() + 1)

    # Among the optimal partition's units, the count is the size of the unit
    # with the greatest compression gain (the dominant repeated structure).
    best_idx: List[int] = []
    best_gain = -np.inf
    for c in np.unique(labels):
        idx = np.where(labels == c)[0].tolist()
        gain = _unit_compression_gain(candidates, idx, keys, n)
        if gain > best_gain:
            best_gain = gain
            best_idx = idx

    return {
        "count": len(best_idx),
        "k_star": k_units,
        "cuts": k_star_cuts,
        "dominant_gain": float(best_gain),
        "indices": best_idx,
        "curve": curve,
    }


def mdl_count_group(candidates: List[Candidate]) -> CountGroup:
    """Convenience wrapper returning the dominant unit as a CountGroup."""
    out = mdl_select_count(candidates)
    return CountGroup(
        indices=out["indices"],
        count=out["count"],
        group_type="object_or_counting_unit",
        meta={k: out[k] for k in ("k_star", "cuts", "dominant_gain")},
    )
