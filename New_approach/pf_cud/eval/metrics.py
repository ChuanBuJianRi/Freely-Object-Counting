"""Counting metrics."""

from typing import List

import numpy as np


def mae(pred_counts: List[int], gt_counts: List[int]) -> float:
    return float(np.mean([abs(p - g) for p, g in zip(pred_counts, gt_counts)]))


def rmse(pred_counts: List[int], gt_counts: List[int]) -> float:
    return float(np.sqrt(np.mean([(p - g) ** 2 for p, g in zip(pred_counts, gt_counts)])))


def nae(pred_counts: List[int], gt_counts: List[int]) -> float:
    vals = []
    for p, g in zip(pred_counts, gt_counts):
        vals.append(abs(p - g) / max(1, g))
    return float(np.mean(vals))


def sre(pred_counts: List[int], gt_counts: List[int]) -> float:
    vals = []
    for p, g in zip(pred_counts, gt_counts):
        vals.append(((p - g) ** 2) / max(1, g))
    return float(np.mean(vals))
