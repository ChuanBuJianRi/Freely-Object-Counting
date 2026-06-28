"""SAM2 自动候选生成（OV-CUD §6.1 + §6.3）。

用冻结的 SAM2（transformers mask-generation pipeline）生成过完备 mask 候选，
再做轻量过滤：去掉过小/过大 mask、近重复 mask，并限制候选上限。

SAM2 candidate != 真实 instance，后续靠 relation / refinement / representative 处理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    SAM2_MODEL,
    SAM2_POINTS_PER_BATCH,
    Sam2Config,
)
from ..candidates.geometry import mask_iou


@dataclass
class Candidate:
    mask: np.ndarray                       # H x W uint8
    bbox: Tuple[int, int, int, int]        # [x1, y1, x2, y2]
    area: float
    source_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def _mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class SAM2Proposer:
    """封装 SAM2 mask-generation pipeline，输出过滤后的候选列表。"""

    def __init__(
        self,
        model_name: str = SAM2_MODEL,
        device: str = "cpu",
        points_per_batch: int = SAM2_POINTS_PER_BATCH,
        config: Optional[Sam2Config] = None,
    ) -> None:
        from transformers import pipeline

        # config 优先；否则用显式参数构造一份默认配置以保持向后兼容
        self.config = config or Sam2Config(
            model_name=model_name,
            device=device,
            points_per_batch=points_per_batch,
        )
        cfg = self.config

        dev = -1 if cfg.device == "cpu" else 0
        self.generator = pipeline("mask-generation", model=cfg.model_name, device=dev)
        self.points_per_batch = cfg.points_per_batch

    def generate(
        self,
        image: np.ndarray,
        max_candidates: int = 0,
    ) -> List[Candidate]:
        """对单张图像生成候选。image: H x W x 3 uint8。

        max_candidates <= 0 表示无上限（OCCAM-M 配方，对齐 config.max_candidates_per_image=0）。
        """
        from PIL import Image

        cfg = self.config
        h, w = image.shape[:2]
        img_area = float(h * w)
        pil = Image.fromarray(image).convert("RGB")

        out = self.generator(pil, **cfg.amg_kwargs())
        masks = out["masks"]
        scores = out.get("scores", None)
        if scores is not None:
            scores = [float(s) for s in scores]
        else:
            scores = [0.0] * len(masks)

        raw: List[Candidate] = []
        for m, s in zip(masks, scores):
            m = np.asarray(m).astype(np.uint8)
            area = float(m.sum())
            if area <= 0:
                continue
            ar = area / img_area
            if ar < cfg.min_area_ratio or ar > cfg.max_area_ratio:
                continue
            box = _mask_to_bbox(m)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if (x2 - x1) < cfg.min_box_size or (y2 - y1) < cfg.min_box_size:
                continue
            raw.append(Candidate(mask=m, bbox=box, area=area, source_score=s))

        # 按 source_score 降序，贪心去近重复
        raw.sort(key=lambda c: c.source_score, reverse=True)
        kept: List[Candidate] = []
        for c in raw:
            dup = False
            for k in kept:
                # bbox 快速预筛后再算 mask IoU
                if (
                    _box_iou(c.bbox, k.bbox) > cfg.dedup_box_iou_prescreen
                    and mask_iou(c.mask, k.mask) > cfg.near_dup_iou
                ):
                    dup = True
                    break
            if not dup:
                kept.append(c)
            if max_candidates > 0 and len(kept) >= max_candidates:
                break
        return kept


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


__all__ = ["Candidate", "SAM2Proposer"]
