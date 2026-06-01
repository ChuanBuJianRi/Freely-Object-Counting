"""Candidate canonicalization and deduplication.

Multi-source candidates produce many overlapping regions. Instead of a fixed
IoU threshold, we build overlap connected components (overlap is a geometric
fact, IoU > 0, not a tuned threshold) and within each component partition the
duplicates using MST + Otsu, keeping one representative per duplicate group.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from skimage.filters import threshold_otsu

from pf_cud.data import Candidate


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def _bbox_overlaps(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> bool:
    return not (b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1])


def _local_iou(c1: Candidate, c2: Candidate) -> float:
    """IoU computed only over the union bbox region (avoids full-image ops)."""
    x1 = min(c1.bbox[0], c2.bbox[0])
    y1 = min(c1.bbox[1], c2.bbox[1])
    x2 = max(c1.bbox[2], c2.bbox[2])
    y2 = max(c1.bbox[3], c2.bbox[3])
    m1 = c1.mask[y1:y2, x1:x2]
    m2 = c2.mask[y1:y2, x1:x2]
    inter = np.logical_and(m1, m2).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def overlap_iou_edges(
    candidates: List[Candidate],
) -> Dict[Tuple[int, int], float]:
    """Sparse IoU distances only for bbox-intersecting candidate pairs.

    Overlap (IoU > 0) is a geometric fact, not a tuned threshold. We use a
    bbox-interval sweep to find intersecting pairs, then compute IoU locally.
    """
    n = len(candidates)
    boxes = [c.bbox for c in candidates]

    # Sort candidates by x1 and sweep; only compare with those whose x-interval
    # still overlaps. This avoids the full O(n^2) bbox check for sparse layouts.
    order = sorted(range(n), key=lambda i: boxes[i][0])
    edges: Dict[Tuple[int, int], float] = {}

    active: List[int] = []
    for idx in order:
        bx = boxes[idx]
        # drop candidates whose x2 is left of current x1.
        active = [k for k in active if boxes[k][2] > bx[0]]
        for k in active:
            if _bbox_overlaps(boxes[k], bx):
                iou = _local_iou(candidates[k], candidates[idx])
                if iou > 0.0:
                    a, b = (k, idx) if k < idx else (idx, k)
                    edges[(a, b)] = 1.0 - iou
        active.append(idx)

    return edges


def auto_partition_by_mst(distance_matrix: np.ndarray) -> List[List[int]]:
    """对任意距离矩阵执行 MST -> Otsu cut -> connected components。

    不需要 k、epsilon、delta。
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst = mst + mst.T

    edge_values = mst[mst > 0]
    if len(edge_values) == 0:
        return [[i] for i in range(n)]

    if len(np.unique(edge_values)) == 1:
        return [list(range(n))]

    tau = threshold_otsu(edge_values)

    kept = (mst > 0) & (mst <= tau)
    graph = kept.astype(np.int32)

    n_components, labels = connected_components(graph, directed=False)

    groups = []
    for c in range(n_components):
        groups.append(np.where(labels == c)[0].tolist())

    return groups


def choose_representative(candidates: List[Candidate], indices: List[int]) -> Candidate:
    """重复候选中选择代表，不用人工规则阈值。

    - 优先选择 mask area 接近 group median area 的候选；
    - 如果并列，优先 SAM；
    - 如果再并列，选择 score 更高者。
    """
    areas = np.array([candidates[i].mask.sum() for i in indices], dtype=np.float64)
    med = np.median(areas)
    area_rank = np.abs(areas - med)

    source_bonus = np.array(
        [0.0 if candidates[i].source == "sam" else 1.0 for i in indices]
    )

    scores = np.array(
        [-(candidates[i].score if candidates[i].score is not None else 0.0) for i in indices]
    )

    order = np.lexsort((scores, source_bonus, area_rank))
    return candidates[indices[int(order[0])]]


def deduplicate_candidates(candidates: List[Candidate]) -> List[Candidate]:
    n = len(candidates)
    if n <= 1:
        return candidates

    edges = overlap_iou_edges(candidates)

    # Overlap connected components via sparse graph.
    if edges:
        rows = [a for (a, b) in edges]
        cols = [b for (a, b) in edges]
        data = [1 for _ in edges]
        adj = coo_matrix((data, (rows, cols)), shape=(n, n))
        n_comp, labels = connected_components(adj, directed=False)
    else:
        n_comp, labels = n, np.arange(n)

    comp_members: Dict[int, List[int]] = {}
    for i in range(n):
        comp_members.setdefault(int(labels[i]), []).append(i)

    final: List[Candidate] = []
    for comp in comp_members.values():
        if len(comp) == 1:
            final.append(candidates[comp[0]])
            continue

        local = {g: k for k, g in enumerate(comp)}
        m = len(comp)
        sub_d = np.ones((m, m), dtype=np.float64)
        np.fill_diagonal(sub_d, 0.0)
        for (a, b), dist in edges.items():
            if a in local and b in local:
                ia, ib = local[a], local[b]
                sub_d[ia, ib] = dist
                sub_d[ib, ia] = dist

        duplicate_groups_local = auto_partition_by_mst(sub_d)

        for g in duplicate_groups_local:
            global_indices = [comp[idx] for idx in g]
            rep = choose_representative(candidates, global_indices)
            rep.meta["merged_from"] = global_indices
            final.append(rep)

    return final
