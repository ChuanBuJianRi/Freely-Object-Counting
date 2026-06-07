"""Candidate <-> GT matching (design.md sec 10.2).

For each candidate mask M_i and GT instance G_k:
    IoU      = |M_i ^ G_k| / |M_i u G_k|
    purity   = |M_i ^ G_k| / |M_i|
    coverage = |M_i ^ G_k| / |G_k|
Match k* = argmax_k IoU; validity from purity/coverage thresholds.

Pure-numpy, dataset-independent, unit-testable with synthetic masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..config import MatchConfig
from .dataset import GTInstance

VALID = "valid"
PART = "part"
BACKGROUND = "background"


@dataclass
class MatchResult:
    iou: np.ndarray              # [N]
    purity: np.ndarray          # [N]
    coverage: np.ndarray        # [N]
    matched_gt: np.ndarray      # [N] index into gt list, -1 if none
    matched_class: List[str]    # [N], "" if none
    matched_instance_id: np.ndarray  # [N], -1 if none
    validity: List[str]         # [N] in {valid, part, background}
    is_valid: np.ndarray        # [N] bool (valid or part)
    weight: np.ndarray          # [N] purity * is_valid


def match_candidates_to_gt(
    candidate_masks: Sequence[np.ndarray],
    gt: List[GTInstance],
    config: MatchConfig,
) -> MatchResult:
    n = len(candidate_masks)
    if n == 0 or len(gt) == 0:
        return MatchResult(
            iou=np.zeros(n), purity=np.zeros(n), coverage=np.zeros(n),
            matched_gt=np.full(n, -1, dtype=np.int64),
            matched_class=["" for _ in range(n)],
            matched_instance_id=np.full(n, -1, dtype=np.int64),
            validity=[BACKGROUND for _ in range(n)],
            is_valid=np.zeros(n, dtype=bool), weight=np.zeros(n),
        )

    cm = np.stack([m.reshape(-1).astype(np.float32) for m in candidate_masks], axis=0)  # [N, P]
    gm = np.stack([g.mask.reshape(-1).astype(np.float32) for g in gt], axis=0)          # [K, P]
    cand_area = cm.sum(axis=1)        # [N]
    gt_area = gm.sum(axis=1)          # [K]
    inter = cm @ gm.T                 # [N, K]
    union = cand_area[:, None] + gt_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou_mat = np.where(union > 0, inter / union, 0.0)
        purity_mat = np.where(cand_area[:, None] > 0, inter / cand_area[:, None], 0.0)
        coverage_mat = np.where(gt_area[None, :] > 0, inter / gt_area[None, :], 0.0)

    matched_gt = np.argmax(iou_mat, axis=1)
    rows = np.arange(n)
    iou = iou_mat[rows, matched_gt]
    purity = purity_mat[rows, matched_gt]
    coverage = coverage_mat[rows, matched_gt]
    # best purity over ALL gt (sec 10.2 uses max_k purity for the validity gate)
    max_purity = purity_mat.max(axis=1)

    matched_class: List[str] = []
    matched_instance_id = np.full(n, -1, dtype=np.int64)
    validity: List[str] = []
    for i in range(n):
        k = int(matched_gt[i])
        if max_purity[i] < config.tau_purity:
            validity.append(BACKGROUND)
            matched_class.append("")
            matched_gt[i] = -1
            continue
        if coverage[i] < config.tau_part and purity[i] >= config.tau_purity:
            validity.append(PART)
        else:
            validity.append(VALID)
        matched_class.append(gt[k].class_name)
        matched_instance_id[i] = gt[k].instance_id

    is_valid = np.array([v in (VALID, PART) for v in validity], dtype=bool)
    weight = purity * is_valid.astype(np.float32)

    return MatchResult(
        iou=iou, purity=purity, coverage=coverage, matched_gt=matched_gt,
        matched_class=matched_class, matched_instance_id=matched_instance_id,
        validity=validity, is_valid=is_valid, weight=weight,
    )
