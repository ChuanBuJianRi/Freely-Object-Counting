"""Hungarian matching of predicted vs ground-truth counts."""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_counts(
    pred_counts: List[int], gt_counts: List[int]
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if len(pred_counts) == 0 or len(gt_counts) == 0:
        return [], list(range(len(pred_counts))), list(range(len(gt_counts)))

    cost = np.zeros((len(pred_counts), len(gt_counts)), dtype=np.float64)
    for i, p in enumerate(pred_counts):
        for j, g in enumerate(gt_counts):
            cost[i, j] = abs(p - g)

    row, col = linear_sum_assignment(cost)

    matched = list(zip(row.tolist(), col.tolist()))
    unmatched_pred = sorted(set(range(len(pred_counts))) - set(row.tolist()))
    unmatched_gt = sorted(set(range(len(gt_counts))) - set(col.tolist()))

    return matched, unmatched_pred, unmatched_gt
