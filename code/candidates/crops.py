"""候选裁剪（OV-CUD §6.2 Candidate Canonicalization）。

为每个候选构造三路输入：
    masked crop  : 只保留 mask 内部，背景置零（强化本体外观）
    box crop     : 裁 bbox，不清背景（保留局部上下文）
    context crop : bbox 外扩后裁剪（判断完整性 / 局部-整体 / 邻近实例）
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..config import CONTEXT_EXPAND


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return x1, y1, x2, y2


def build_three_crops(
    image: np.ndarray,
    mask: np.ndarray,
    bbox: Tuple[float, float, float, float],
    context_expand: float = CONTEXT_EXPAND,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """返回 (masked_crop, box_crop, context_crop) 三个 PIL.Image。

    image : H x W x 3 uint8
    mask  : H x W {0,1}
    bbox  : [x1, y1, x2, y2]
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_box(*bbox, w=w, h=h)

    # box crop：直接裁 bbox
    box_crop = image[y1:y2, x1:x2]

    # masked crop：mask 外置零后裁 bbox
    m = mask.astype(bool)
    masked_full = image * m[:, :, None]
    masked_crop = masked_full[y1:y2, x1:x2]

    # context crop：bbox 外扩后裁
    bw, bh = x2 - x1, y2 - y1
    ex, ey = int(bw * context_expand), int(bh * context_expand)
    cx1, cy1, cx2, cy2 = _clip_box(x1 - ex, y1 - ey, x2 + ex, y2 + ey, w=w, h=h)
    context_crop = image[cy1:cy2, cx1:cx2]

    return (
        Image.fromarray(masked_crop).convert("RGB"),
        Image.fromarray(box_crop).convert("RGB"),
        Image.fromarray(context_crop).convert("RGB"),
    )


__all__ = ["build_three_crops"]
