"""Otsu-based MST graph cutting (no delta, no manual threshold)."""

import numpy as np
from skimage.filters import threshold_otsu


def otsu_cut_mst(mst: np.ndarray) -> np.ndarray:
    """输入 MST adjacency matrix，输出 cut 后的 adjacency matrix。"""
    graph = mst.copy()
    edge_values = graph[graph > 0]

    if len(edge_values) == 0:
        return graph

    unique = np.unique(edge_values)
    if len(unique) == 1:
        # 没有自然断点，保留原 MST。
        return graph

    tau = threshold_otsu(edge_values)

    # 大于 tau 的边被认为是跨类长边。
    graph[graph > tau] = 0.0
    return graph
