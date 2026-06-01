"""Hypothesis ranking: separate objects, patterns and background textures."""

from typing import Dict, List

import numpy as np

from pf_cud.data import Candidate, CountGroup


def group_centers(candidates: List[Candidate], group: CountGroup) -> np.ndarray:
    centers = []
    h, w = candidates[0].mask.shape[:2]
    for idx in group.indices:
        x1, y1, x2, y2 = candidates[idx].bbox
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        centers.append([cx, cy])
    return np.array(centers, dtype=np.float64)


def group_areas(candidates: List[Candidate], group: CountGroup) -> np.ndarray:
    h, w = candidates[0].mask.shape[:2]
    return np.array(
        [candidates[idx].mask.sum() / float(h * w) for idx in group.indices],
        dtype=np.float64,
    )


def feature_residual(candidates: List[Candidate], group: CountGroup, key: str) -> float:
    if len(group.indices) <= 1:
        return float("inf")
    if not all(key in candidates[i].features for i in group.indices):
        return 0.0

    x = np.stack([candidates[i].features[key] for i in group.indices], axis=0)
    mu = x.mean(axis=0, keepdims=True)
    return float(np.mean((x - mu) ** 2))


def compute_group_raw_scores(
    candidates: List[Candidate], group: CountGroup
) -> Dict[str, float]:
    centers = group_centers(candidates, group)
    areas = group_areas(candidates, group)

    count = len(group.indices)

    repeatability = np.log1p(count)

    center_dists = np.sqrt(((centers - 0.5) ** 2).sum(axis=1))
    centrality = -float(np.mean(center_dists))

    area_consistency = -float(np.std(areas) / (np.mean(areas) + 1e-12))

    visual_consistency = -feature_residual(candidates, group, "visual")
    shape_consistency = -feature_residual(candidates, group, "shape")
    color_consistency = -feature_residual(candidates, group, "color")

    spatial_spread = (
        float(np.linalg.det(np.cov(centers.T) + np.eye(2) * 1e-6))
        if len(centers) > 1
        else 0.0
    )
    backgroundness = spatial_spread + repeatability - abs(centrality)

    return {
        "repeatability": repeatability,
        "centrality": centrality,
        "area_consistency": area_consistency,
        "visual_consistency": visual_consistency,
        "shape_consistency": shape_consistency,
        "color_consistency": color_consistency,
        "backgroundness": backgroundness,
    }


def rank_values(values: List[float], higher_is_better: bool = True) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    # 把 inf/-inf 替换为有限极值，避免排序不稳定。
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        hi = finite.max()
        lo = finite.min()
    else:
        hi, lo = 0.0, 0.0
    arr = np.where(np.isposinf(arr), hi, arr)
    arr = np.where(np.isneginf(arr), lo, arr)

    order = np.argsort(arr)
    if higher_is_better:
        order = order[::-1]

    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)

    if len(arr) > 1:
        ranks = 1.0 - ranks / (len(arr) - 1)
    else:
        ranks = np.ones_like(ranks, dtype=np.float64)

    return ranks


def rank_groups(
    candidates: List[Candidate], groups: List[CountGroup]
) -> List[CountGroup]:
    if not groups:
        return groups

    raw = [compute_group_raw_scores(candidates, g) for g in groups]

    keys_for_main_object = [
        "repeatability",
        "centrality",
        "area_consistency",
        "visual_consistency",
        "shape_consistency",
        "color_consistency",
    ]

    rank_mats = []
    for key in keys_for_main_object:
        vals = [r[key] for r in raw]
        rank_mats.append(rank_values(vals, higher_is_better=True))

    main_scores = np.mean(np.stack(rank_mats, axis=0), axis=0)

    bg_rank = rank_values([r["backgroundness"] for r in raw], higher_is_better=True)

    for i, g in enumerate(groups):
        g.score = float(main_scores[i])
        g.confidence = float(main_scores[i])

        if bg_rank[i] == bg_rank.max() and len(groups) > 1:
            g.group_type = "background_or_pattern"
        else:
            g.group_type = "object_or_counting_unit"

        g.meta["raw_scores"] = raw[i]
        g.meta["main_rank_score"] = float(main_scores[i])
        g.meta["background_rank_score"] = float(bg_rank[i])

    groups = sorted(
        groups, key=lambda g: g.score if g.score is not None else -1, reverse=True
    )
    return groups
