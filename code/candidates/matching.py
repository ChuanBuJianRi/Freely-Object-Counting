"""候选-GT 匹配（OV-CUD §10.2 / 2_training_plan.md §1.3）。

对每个 SAM2 候选 M_i 与所有 GT 实例 G_k 计算 IoU / purity / coverage，
按 IoU 取最佳匹配 k*，并据有效性规则打出训练标签：

    purity_i              = purity_{i,k*}
    coverage_i            = coverage_{i,k*}
    matched_class_i       = class(G_k*)
    matched_instance_id_i = id(G_k*)
    valid_i / label_type  = 由 tau_purity / tau_part 决定

有效性规则（与文档一致）：
    max_k purity < tau_purity                       -> background/noise（valid=0，不作强正样本）
    coverage < tau_part 且 purity 高                -> part candidate（valid=1，权重在损失里按 purity 体现）
    else                                            -> valid semantic candidate（valid=1）

低 purity 候选绝不硬分配为强正样本（valid=0），避免污染 Category / Relation 标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..config import TAU_PART, TAU_PURITY
from .geometry import mask_overlaps

# label_type 取值
LABEL_BACKGROUND = "background"   # 低 purity，valid=0
LABEL_PART = "part"               # 高 purity 低 coverage，valid=1（部件）
LABEL_VALID = "valid"             # 正常语义候选，valid=1


@dataclass
class MatchResult:
    """单个候选的匹配结果。"""

    purity: float
    coverage: float
    iou: float
    matched_class: int            # contiguous class idx；background 时为 -1
    matched_instance_id: int      # GT 实例 id；background 时为 -1
    valid: int                    # 0/1，是否作为类别正样本
    label_type: str               # background / part / valid
    matched_gt_index: int = -1    # 命中的 GT 在输入列表中的下标（背景为 -1）

    def as_dict(self) -> Dict[str, float]:
        return {
            "purity": self.purity,
            "coverage": self.coverage,
            "iou": self.iou,
            "matched_class": self.matched_class,
            "matched_instance_id": self.matched_instance_id,
            "valid": self.valid,
            "label_type": self.label_type,
            "matched_gt_index": self.matched_gt_index,
        }


@dataclass
class GTInstance:
    """匹配所需的最小 GT 字段（与 data.coco_lvis 的 instance dict 对齐）。"""

    mask: np.ndarray
    class_idx: int
    instance_id: int


def _to_gt_instances(instances: Sequence[Dict]) -> List[GTInstance]:
    out: List[GTInstance] = []
    for ins in instances:
        out.append(
            GTInstance(
                mask=np.asarray(ins["mask"]).astype(bool),
                class_idx=int(ins["class_idx"]),
                instance_id=int(ins["instance_id"]),
            )
        )
    return out


def match_candidate(
    cand_mask: np.ndarray,
    gts: Sequence[GTInstance],
    tau_purity: float = TAU_PURITY,
    tau_part: float = TAU_PART,
) -> MatchResult:
    """对单个候选 mask 与一组 GT 实例做匹配，返回 MatchResult。"""
    cand = np.asarray(cand_mask).astype(bool)

    if not gts:
        return MatchResult(
            purity=0.0, coverage=0.0, iou=0.0,
            matched_class=-1, matched_instance_id=-1,
            valid=0, label_type=LABEL_BACKGROUND, matched_gt_index=-1,
        )

    best_iou = -1.0
    best_k = -1
    best_purity = 0.0
    best_coverage = 0.0
    max_purity = 0.0   # 跨所有 GT 的最大 purity，用于 background 判定
    for k, g in enumerate(gts):
        iou, purity, coverage = mask_overlaps(cand, g.mask)
        max_purity = max(max_purity, purity)
        if iou > best_iou:
            best_iou = iou
            best_k = k
            best_purity = purity
            best_coverage = coverage

    # 有效性规则
    if max_purity < tau_purity:
        return MatchResult(
            purity=best_purity, coverage=best_coverage, iou=max(best_iou, 0.0),
            matched_class=-1, matched_instance_id=-1,
            valid=0, label_type=LABEL_BACKGROUND, matched_gt_index=-1,
        )

    g = gts[best_k]
    label_type = LABEL_PART if (best_coverage < tau_part and best_purity >= tau_purity) else LABEL_VALID
    return MatchResult(
        purity=best_purity, coverage=best_coverage, iou=max(best_iou, 0.0),
        matched_class=g.class_idx, matched_instance_id=g.instance_id,
        valid=1, label_type=label_type, matched_gt_index=best_k,
    )


def match_candidates(
    cand_masks: Sequence[np.ndarray],
    instances: Sequence[Dict],
    tau_purity: float = TAU_PURITY,
    tau_part: float = TAU_PART,
) -> List[MatchResult]:
    """批量匹配：candidate masks vs GT instances。

    instances: data.coco_lvis 的 instance dict 列表，含 mask/class_idx/instance_id。
    """
    gts = _to_gt_instances(instances)
    return [match_candidate(m, gts, tau_purity, tau_part) for m in cand_masks]


def stack_match_labels(results: Sequence[MatchResult]) -> Dict[str, np.ndarray]:
    """把一组 MatchResult 堆叠成 numpy 数组，便于写入缓存。"""
    n = len(results)
    return {
        "purity": np.array([r.purity for r in results], dtype=np.float32),
        "coverage": np.array([r.coverage for r in results], dtype=np.float32),
        "iou": np.array([r.iou for r in results], dtype=np.float32),
        "matched_class": np.array([r.matched_class for r in results], dtype=np.int64),
        "matched_instance_id": np.array([r.matched_instance_id for r in results], dtype=np.int64),
        "valid": np.array([r.valid for r in results], dtype=np.int64),
        "matched_gt_index": np.array([r.matched_gt_index for r in results], dtype=np.int64),
    }


__all__ = [
    "MatchResult",
    "GTInstance",
    "match_candidate",
    "match_candidates",
    "stack_match_labels",
    "LABEL_BACKGROUND",
    "LABEL_PART",
    "LABEL_VALID",
]
