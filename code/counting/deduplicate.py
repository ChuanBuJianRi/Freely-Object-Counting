"""Same-Instance Deduplication（OV-CUD §13.2）。

在每个 semantic group 内，根据 A_inst 构建 same-instance components。
每个 component 最多贡献一个 count。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def build_same_instance_components(
    group_indices: List[int],
    A_inst: np.ndarray,          # [N, N] same-instance relation scores (logits)
    tau_inst: float = 0.5,
) -> List[List[int]]:
    """在 group 内构建 same-instance components。

    两个 candidate 属于同一 instance 当且仅当 sigmoid(A_inst[i,j]) >= tau_inst。
    用 Union-Find 做连通分量。

    Args:
        group_indices: 该 group 内的 candidate 索引列表
        A_inst: [N, N] same-instance 关系 logits
        tau_inst: sigmoid 阈值

    Returns:
        components: list of component index lists（每个 component 是 group_indices 的子集）
    """
    n = len(group_indices)
    if n == 0:
        return []
    if n == 1:
        return [list(group_indices)]

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a in range(n):
        for b in range(a + 1, n):
            gi, gj = group_indices[a], group_indices[b]
            score = float(1.0 / (1.0 + np.exp(-A_inst[gi, gj])))
            if score >= tau_inst:
                union(a, b)

    # 收集 components
    root_to_members = {}
    for i in range(n):
        r = find(i)
        root_to_members.setdefault(r, []).append(group_indices[i])

    return list(root_to_members.values())


__all__ = ["build_same_instance_components"]
