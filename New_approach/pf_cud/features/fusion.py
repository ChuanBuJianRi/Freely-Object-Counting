"""Rank-normalized distance fusion (no manual feature weights)."""

from typing import List

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pf_cud.data import Candidate

_DEFAULT_KEYS = ["visual", "shape", "color", "spatial"]


def pairwise_feature_distance(
    candidates: List[Candidate], key: str, metric: str = "euclidean"
) -> np.ndarray:
    feats = np.stack([cand.features[key] for cand in candidates], axis=0)
    n = len(feats)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    d = squareform(pdist(feats, metric=metric))
    return d.astype(np.float64)


def rank_normalize_distance(d: np.ndarray) -> np.ndarray:
    """把距离矩阵转换成 [0, 1] rank（避免 min-max 受 outlier 影响）。"""
    n = d.shape[0]
    if n <= 1:
        return d.copy()

    tri = np.triu_indices(n, k=1)
    values = d[tri]

    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)

    if len(values) > 1:
        ranks = ranks / (len(values) - 1)
    else:
        ranks = np.zeros_like(ranks, dtype=np.float64)

    out = np.zeros_like(d, dtype=np.float64)
    out[tri] = ranks
    out = out + out.T

    return out


def fused_distance(candidates: List[Candidate]) -> np.ndarray:
    """无权重距离融合：每种 feature 等价参与，但用 rank 平均而非数值平均。

    只融合所有候选都存在且非退化（存在非零距离）的特征。这样像 NullVisual 这种
    常量特征不会拉低有效特征的贡献。
    """
    n = len(candidates)
    if n <= 1:
        return np.zeros((n, n), dtype=np.float64)

    rank_mats = []
    for key in _DEFAULT_KEYS:
        if not all(key in cand.features for cand in candidates):
            continue
        d = pairwise_feature_distance(candidates, key)
        if not np.any(d > 0):
            # 退化特征（全相同），跳过以免稀释。
            continue
        rank_mats.append(rank_normalize_distance(d))

    if not rank_mats:
        return np.zeros((n, n), dtype=np.float64)

    fused = np.mean(np.stack(rank_mats, axis=0), axis=0)
    np.fill_diagonal(fused, 0.0)
    return fused
