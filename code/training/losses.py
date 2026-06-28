"""Category Head 的损失（OV-CUD Stage 1，对应 2_training_plan.md §2.2）。

    L_category = L_cls + lambda_count * L_count + lambda_align * L_align

    - L_cls   : 加权交叉熵，w_i = purity_i * valid_i
    - L_count : is_countable 二分类 BCE
    - L_align : 可选 InfoNCE 视觉-文本对齐（强化 open-vocabulary）

ignore 样本（低 purity）通过 weight=0 自动排除，无需特殊 label。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def weighted_cross_entropy(
    logits: torch.Tensor,        # [B, C]
    targets: torch.Tensor,       # [B]
    sample_weight: torch.Tensor, # [B]  w_i = purity_i * valid_i
    class_weight: Optional[torch.Tensor] = None,  # [C] 长尾平衡，可选
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits, targets, weight=class_weight,
        reduction="none", label_smoothing=label_smoothing,
    )
    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * ce).sum() / denom


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
    gamma: float = 2.0,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    loss = -((1 - pt) ** gamma) * log_pt
    if class_weight is not None:
        loss = loss * class_weight[targets]
    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * loss).sum() / denom


def weighted_bce_with_logits(
    logit: torch.Tensor,         # [B]
    target: torch.Tensor,        # [B] in {0,1}
    sample_weight: torch.Tensor, # [B]
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, target.float(), reduction="none")
    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * bce).sum() / denom


def info_nce_align(
    h: torch.Tensor,                 # [B, D] 已 normalize 的视觉 embedding
    text_prototypes: torch.Tensor,   # [C, D]
    targets: torch.Tensor,           # [B]
    sample_weight: torch.Tensor,     # [B]
    temperature: float = 0.07,
) -> torch.Tensor:
    t = F.normalize(text_prototypes, dim=-1)
    logits = (h @ t.t()) / temperature
    ce = F.cross_entropy(logits, targets, reduction="none")
    denom = sample_weight.sum().clamp_min(1e-6)
    return (sample_weight * ce).sum() / denom


@dataclass
class CategoryLossConfig:
    use_focal: bool = False
    focal_gamma: float = 2.0
    lambda_count: float = 0.5
    lambda_align: float = 0.0       # >0 时启用 InfoNCE 对齐
    align_temperature: float = 0.07
    label_smoothing: float = 0.0    # 仅对非 focal 的加权 CE 生效


def category_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sample_weight: torch.Tensor,
    cfg: CategoryLossConfig,
    *,
    is_countable_logit: Optional[torch.Tensor] = None,
    is_countable_target: Optional[torch.Tensor] = None,
    h: Optional[torch.Tensor] = None,
    text_prototypes: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 (总损失, 各分项字典) 便于日志记录。"""
    if cfg.use_focal:
        l_cls = focal_loss(logits, targets, sample_weight, cfg.focal_gamma, class_weight)
    else:
        l_cls = weighted_cross_entropy(
            logits, targets, sample_weight, class_weight,
            label_smoothing=cfg.label_smoothing,
        )

    parts = {"L_cls": l_cls}
    total = l_cls

    if is_countable_logit is not None and is_countable_target is not None:
        l_count = weighted_bce_with_logits(is_countable_logit, is_countable_target, sample_weight)
        parts["L_count"] = l_count
        total = total + cfg.lambda_count * l_count

    if cfg.lambda_align > 0 and h is not None and text_prototypes is not None:
        l_align = info_nce_align(h, text_prototypes, targets, sample_weight, cfg.align_temperature)
        parts["L_align"] = l_align
        total = total + cfg.lambda_align * l_align

    parts["L_total"] = total
    return total, parts


__all__ = [
    "weighted_cross_entropy",
    "focal_loss",
    "weighted_bce_with_logits",
    "info_nce_align",
    "CategoryLossConfig",
    "category_loss",
]
