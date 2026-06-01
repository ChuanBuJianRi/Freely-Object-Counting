"""Group minimum-description-length (MDL) score."""

from typing import List

import numpy as np

from pf_cud.data import Candidate, CountGroup

_KEYS = ["visual", "shape", "color", "spatial"]


def _group_keys(candidates: List[Candidate], group: CountGroup) -> List[str]:
    if not group.indices:
        return []
    return [
        key
        for key in _KEYS
        if all(key in candidates[i].features for i in group.indices)
    ]


def stack_group_features(
    candidates: List[Candidate], group: CountGroup, key: str
) -> np.ndarray:
    return np.stack([candidates[i].features[key] for i in group.indices], axis=0)


def gaussian_residual_code_length(x: np.ndarray) -> float:
    """用高斯残差近似描述长度，方差由数据自己估计。"""
    if len(x) <= 1:
        return 0.0

    mu = x.mean(axis=0, keepdims=True)
    residual = x - mu

    var = np.mean(residual ** 2) + 1e-12

    n, d = x.shape
    return 0.5 * n * d * np.log(var) + 0.5 * np.sum(residual ** 2) / var


def prototype_cost(x: np.ndarray) -> float:
    """prototype 的复杂度，维度越高成本越高（编码长度的自然项）。"""
    if x.ndim != 2:
        return 0.0
    _, d = x.shape
    return float(d * np.log(2.0 + x.shape[0]))


def group_mdl(candidates: List[Candidate], group: CountGroup) -> float:
    """一个 group 的总描述长度，各特征不使用人工权重直接求和。"""
    if len(group.indices) == 0:
        return 0.0

    total = 0.0
    for key in _group_keys(candidates, group):
        x = stack_group_features(candidates, group, key)
        total += prototype_cost(x)
        total += gaussian_residual_code_length(x)

    # group membership 的编码成本，数量越多成本自然增加。
    total += len(group.indices) * np.log(2.0 + len(candidates))

    return float(total)


def total_mdl(candidates: List[Candidate], groups: List[CountGroup]) -> float:
    return float(sum(group_mdl(candidates, g) for g in groups))


# --- Sufficient-statistics fast path (numerically equivalent to group_mdl) ---
#
# gaussian_residual_code_length uses a single scalar variance over the whole
# (n, d) residual block, so a group's MDL depends only on, per feature key:
#   n         : member count
#   S1[key]   : sum of feature vectors        (shape d,)
#   S2[key]   : sum of squared entries        (scalar)
# Merging two groups just adds these statistics, turning each merge-gain
# evaluation from O(n*d) restacking into O(d).


def common_feature_keys(candidates: List[Candidate]) -> List[str]:
    if not candidates:
        return []
    return [k for k in _KEYS if all(k in c.features for c in candidates)]


def group_stats(candidates: List[Candidate], indices: List[int], keys: List[str]):
    n = len(indices)
    s1 = {}
    s2 = {}
    for key in keys:
        x = np.stack([candidates[i].features[key] for i in indices], axis=0)
        s1[key] = x.sum(axis=0)
        s2[key] = float(np.sum(x ** 2))
    return {"n": n, "s1": s1, "s2": s2}


def merge_stats(a: dict, b: dict, keys: List[str]) -> dict:
    s1 = {key: a["s1"][key] + b["s1"][key] for key in keys}
    s2 = {key: a["s2"][key] + b["s2"][key] for key in keys}
    return {"n": a["n"] + b["n"], "s1": s1, "s2": s2}


def mdl_from_stats(stats: dict, num_candidates: int, keys: List[str]) -> float:
    n = stats["n"]
    if n == 0:
        return 0.0

    total = 0.0
    for key in keys:
        s1 = stats["s1"][key]
        s2 = stats["s2"][key]
        d = int(s1.shape[0])
        total += d * np.log(2.0 + n)  # prototype_cost
        if n > 1:
            sse = s2 - float(np.dot(s1, s1)) / n
            sse = max(sse, 0.0)
            var = sse / (n * d) + 1e-12
            total += 0.5 * n * d * np.log(var) + 0.5 * sse / var

    total += n * np.log(2.0 + num_candidates)
    return float(total)
