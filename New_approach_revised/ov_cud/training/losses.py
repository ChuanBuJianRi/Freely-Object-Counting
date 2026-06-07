"""Weighted losses (design.md sec 10.3, 10.4). Torch-based (training only)."""

from __future__ import annotations


def weighted_cross_entropy(logits, target, weight, ignore_index: int = -1):
    """Sum_i w_i * CE(logits_i, target_i), normalized by total weight.

    logits: [N, C], target: [N] long, weight: [N] float. Entries with
    target == ignore_index are dropped.
    """
    import torch
    import torch.nn.functional as F

    keep = target != ignore_index
    if keep.sum() == 0:
        return logits.sum() * 0.0
    logits, target, weight = logits[keep], target[keep], weight[keep]
    ce = F.cross_entropy(logits, target, reduction="none")
    denom = weight.sum().clamp_min(1e-6)
    return (weight * ce).sum() / denom


def weighted_bce_with_logits(logits, target, weight):
    """Sum_p w_p * BCE(sigmoid(logits_p), target_p), normalized by total weight."""
    import torch
    import torch.nn.functional as F

    if logits.numel() == 0:
        return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    denom = weight.sum().clamp_min(1e-6)
    return (weight * bce).sum() / denom
