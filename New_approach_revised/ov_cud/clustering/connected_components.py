"""Connected components via union-find."""

from __future__ import annotations

from typing import List

import numpy as np


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def connected_components(adjacency: np.ndarray) -> List[List[int]]:
    """Group node indices by connected component of a symmetric 0/1 adjacency."""
    n = adjacency.shape[0]
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j]:
                dsu.union(i, j)
    groups: dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    return list(groups.values())
