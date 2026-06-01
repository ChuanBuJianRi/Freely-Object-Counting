"""Minimum spanning tree construction (no k, no epsilon)."""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def build_mst(distance_matrix: np.ndarray) -> np.ndarray:
    """返回对称 MST adjacency matrix，matrix[i, j] = edge weight or 0。"""
    if distance_matrix.shape[0] <= 1:
        return np.zeros_like(distance_matrix)

    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst = mst + mst.T
    return mst.astype(np.float64)
