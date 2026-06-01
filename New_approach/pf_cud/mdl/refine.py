"""MDL-guided group refinement (greedy merge + optional split)."""

import heapq
from typing import List

import numpy as np

from pf_cud.data import Candidate, CountGroup
from pf_cud.features.fusion import fused_distance
from pf_cud.graph.components import graph_to_groups
from pf_cud.graph.cut import otsu_cut_mst
from pf_cud.graph.mst import build_mst
from pf_cud.mdl.score import (
    common_feature_keys,
    group_mdl,
    group_stats,
    mdl_from_stats,
    merge_stats,
)


def merge_two_groups(a: CountGroup, b: CountGroup) -> CountGroup:
    return CountGroup(
        indices=sorted(a.indices + b.indices),
        count=len(a.indices) + len(b.indices),
    )


def mdl_merge_refinement(
    candidates: List[Candidate], groups: List[CountGroup]
) -> List[CountGroup]:
    """贪心 MDL merge，不需要 merge threshold，只比较 new_mdl < old_mdl。

    用充分统计量评估合并增益，用 lazy-deletion max-heap 取全局最大 gain，
    每次合并只把新组与存活组的成对增益压入堆。语义等价于朴素贪心：每步选
    全局最大 gain 的一对合并，直到没有 gain > 0。
    """
    groups = groups[:]
    n = len(groups)
    if n <= 1:
        for g in groups:
            g.count = len(g.indices)
        return groups

    keys = common_feature_keys(candidates)
    num_c = len(candidates)

    stats = [group_stats(candidates, g.indices, keys) for g in groups]
    self_mdl = [mdl_from_stats(s, num_c, keys) for s in stats]
    alive = [True] * n

    def pair_gain(i: int, j: int) -> float:
        merged = merge_stats(stats[i], stats[j], keys)
        return self_mdl[i] + self_mdl[j] - mdl_from_stats(merged, num_c, keys)

    # Max-heap via negative gain. Entries are lazily invalidated.
    heap = []
    for i in range(n):
        for j in range(i + 1, n):
            g = pair_gain(i, j)
            if g > 0.0:
                heapq.heappush(heap, (-g, i, j))

    while heap:
        neg_g, i, j = heapq.heappop(heap)
        if not alive[i] or not alive[j]:
            continue
        if -neg_g <= 0.0:
            break

        merged_group = merge_two_groups(groups[i], groups[j])
        new_idx = len(groups)
        groups.append(merged_group)
        stats.append(merge_stats(stats[i], stats[j], keys))
        self_mdl.append(mdl_from_stats(stats[new_idx], num_c, keys))
        alive[i] = False
        alive[j] = False
        alive.append(True)

        for k in range(new_idx):
            if alive[k]:
                g = pair_gain(k, new_idx)
                if g > 0.0:
                    heapq.heappush(heap, (-g, k, new_idx))

    result = [groups[k] for k in range(len(groups)) if alive[k]]
    for g in result:
        g.count = len(g.indices)
    return result


def _split_group_once(
    candidates: List[Candidate], group: CountGroup
) -> List[CountGroup]:
    """对单个 group 内部再跑 MST + Otsu，得到候选拆分。"""
    if len(group.indices) <= 1:
        return [group]

    sub = [candidates[i] for i in group.indices]
    d = fused_distance(sub)
    mst = build_mst(d)
    cut = otsu_cut_mst(mst)
    local_groups = graph_to_groups(cut)

    if len(local_groups) <= 1:
        return [group]

    result: List[CountGroup] = []
    for lg in local_groups:
        global_indices = [group.indices[k] for k in lg.indices]
        result.append(CountGroup(indices=sorted(global_indices), count=len(global_indices)))
    return result


def mdl_split_refinement(
    candidates: List[Candidate], groups: List[CountGroup]
) -> List[CountGroup]:
    """对每个 group 尝试拆分，若拆分后 total MDL 更低则接受（无阈值）。"""
    refined: List[CountGroup] = []
    for g in groups:
        candidate_split = _split_group_once(candidates, g)
        if len(candidate_split) <= 1:
            refined.append(g)
            continue

        old_score = group_mdl(candidates, g)
        new_score = sum(group_mdl(candidates, s) for s in candidate_split)
        if new_score < old_score:
            refined.extend(candidate_split)
        else:
            refined.append(g)

    for g in refined:
        g.count = len(g.indices)
    return refined
