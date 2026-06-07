"""Lightweight candidate filtering (design.md sec 6.3).

Only removes obvious noise (too small / too large / low score / near-duplicate).
It does NOT replace the relation head. Uncertain-but-plausible repeated units
are intentionally kept for later unknown/pattern handling.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import FilterConfig
from ..data import Candidate


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def filter_candidates(
    candidates: List[Candidate], image_shape, config: FilterConfig
) -> List[Candidate]:
    h, w = image_shape[:2]
    img_area = float(max(1, h * w))

    # 1) size + score gate.
    kept: List[Candidate] = []
    for c in candidates:
        ratio = c.area / img_area
        if ratio < config.min_area_ratio or ratio > config.max_area_ratio:
            continue
        if c.source_score is not None and c.source_score < config.min_source_score:
            continue
        kept.append(c)

    # 2) near-duplicate suppression: keep higher source_score on high IoU pairs.
    order = sorted(
        range(len(kept)),
        key=lambda i: (kept[i].source_score or 0.0, kept[i].area),
        reverse=True,
    )
    survivors: List[int] = []
    for i in order:
        dup = False
        for j in survivors:
            if _iou(kept[i].mask, kept[j].mask) >= config.dedup_iou:
                dup = True
                break
        if not dup:
            survivors.append(i)
    kept = [kept[i] for i in survivors]

    # 3) hard cap on candidate count (full N^2 downstream).
    if config.max_candidates is not None and len(kept) > config.max_candidates:
        kept = sorted(kept, key=lambda c: (c.source_score or 0.0), reverse=True)
        kept = kept[: config.max_candidates]

    return kept
