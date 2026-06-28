"""候选几何特征与 mask 集合运算（OV-CUD §6.2 / §10.2）。

提供：
    - geometry_features: 归一化中心、宽高、面积比、长宽比、紧致度等
    - mask 级 IoU / purity / coverage / containment（候选-GT 匹配与 pairwise 关系共用）
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def geometry_features(
    bbox: Tuple[float, float, float, float],
    mask_area: float,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """返回固定长度几何特征向量（8 维）。"""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    bbox_area = bw * bh
    img_area = float(img_h * img_w)
    return np.array(
        [
            cx,                              # 归一化中心 x
            cy,                              # 归一化中心 y
            nw,                              # 归一化宽
            nh,                              # 归一化高
            bbox_area / img_area,            # bbox 面积占图比
            mask_area / img_area,            # mask 面积占图比
            bw / bh,                         # 长宽比
            mask_area / bbox_area,           # mask/bbox（紧致度/填充率）
        ],
        dtype=np.float32,
    )


def mask_overlaps(cand: np.ndarray, gt: np.ndarray) -> Tuple[float, float, float]:
    """返回 (IoU, purity, coverage)。

        purity   = |cand ∩ gt| / |cand|     候选有多少落在该 GT 内
        coverage = |cand ∩ gt| / |gt|       该 GT 被候选覆盖多少
    """
    c = cand.astype(bool)
    g = gt.astype(bool)
    inter = float(np.logical_and(c, g).sum())
    ac = float(c.sum())
    ag = float(g.sum())
    union = ac + ag - inter
    iou = inter / union if union > 0 else 0.0
    purity = inter / ac if ac > 0 else 0.0
    coverage = inter / ag if ag > 0 else 0.0
    return iou, purity, coverage


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    am, bm = a.astype(bool), b.astype(bool)
    inter = float(np.logical_and(am, bm).sum())
    union = float(np.logical_or(am, bm).sum())
    return inter / union if union > 0 else 0.0


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """|a ∩ b| / |a|：a 有多少被 b 包含。"""
    am, bm = a.astype(bool), b.astype(bool)
    aa = float(am.sum())
    if aa == 0:
        return 0.0
    return float(np.logical_and(am, bm).sum()) / aa


__all__ = ["geometry_features", "mask_overlaps", "mask_iou", "containment"]
