"""候选过滤（OV-CUD §6.3 Candidate Filtering）。

对 SAM2 过完备候选做轻量过滤，去掉明显无效项，避免它们进入全量 pairwise 计算：

    - 过小 / 过大 mask（面积占比越界）
    - 过小 bbox
    - 低 source_score（可选）
    - 高 IoU 近重复（保留 source_score 更高者）

过滤只处理明显噪声，不替代 relation head；不确定但合理的重复单元应保留以支持
unknown / repeated-pattern 处理。

注：SAM2Proposer.generate 已内置同等过滤；本模块把过滤抽成对 Candidate 列表操作
的独立纯函数，供推理管线（pipeline）在任意候选来源上复用。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from ..config import (
    DEDUP_BOX_IOU_PRESCREEN,
    MAX_AREA_RATIO,
    MIN_AREA_RATIO,
    MIN_BOX_SIZE,
    NEAR_DUP_IOU,
)
from .geometry import mask_iou


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def filter_candidates(
    candidates: Sequence,
    img_h: int,
    img_w: int,
    *,
    min_area_ratio: float = MIN_AREA_RATIO,
    max_area_ratio: float = MAX_AREA_RATIO,
    min_box_size: int = MIN_BOX_SIZE,
    min_source_score: float = 0.0,
    near_dup_iou: float = NEAR_DUP_IOU,
    box_iou_prescreen: float = DEDUP_BOX_IOU_PRESCREEN,
    max_candidates: int = 0,
) -> List:
    """过滤候选列表，返回保留的候选（保持 Candidate 类型，不复制 mask）。

    candidates: 任意带 .mask / .bbox / .area / .source_score 字段的对象列表
                （如 proposals.sam2_proposal.Candidate）。
    """
    img_area = float(img_h * img_w)
    if img_area <= 0:
        return []

    # 1) 面积 / bbox / score 过滤
    survived: List = []
    for c in candidates:
        area = float(getattr(c, "area", float(np.asarray(c.mask).sum())))
        if area <= 0:
            continue
        ar = area / img_area
        if ar < min_area_ratio or ar > max_area_ratio:
            continue
        x1, y1, x2, y2 = c.bbox
        if (x2 - x1) < min_box_size or (y2 - y1) < min_box_size:
            continue
        if float(getattr(c, "source_score", 0.0)) < min_source_score:
            continue
        survived.append(c)

    # 2) 近重复去重：按 source_score 降序贪心，bbox 预筛后再算 mask IoU
    survived.sort(key=lambda c: float(getattr(c, "source_score", 0.0)), reverse=True)
    kept: List = []
    for c in survived:
        dup = False
        for k in kept:
            if (
                _box_iou(c.bbox, k.bbox) > box_iou_prescreen
                and mask_iou(np.asarray(c.mask), np.asarray(k.mask)) > near_dup_iou
            ):
                dup = True
                break
        if not dup:
            kept.append(c)
        if max_candidates > 0 and len(kept) >= max_candidates:
            break
    return kept


__all__ = ["filter_candidates"]
