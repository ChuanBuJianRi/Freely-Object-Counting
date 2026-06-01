"""Prediction heads for OCCAM counting results.

Given an ``OccamResult`` (clusters + masks) and the original image shape,
return a single integer count. Three strategies are supported:

- ``"total"``: sum of all cluster sizes (paper default; keeps every cluster,
  including singletons / background fragments).
- ``"max"``: largest cluster size (FSC-147-friendly when the query class
  forms one dominant cluster; what the OCCAM-MP7 baseline uses).
- ``"mode_cluster_vote"`` / ``"mcv"``: anchored at the largest cluster, sum
  every other cluster whose log-area is within ``k * MAD`` of the anchor.
  Designed to keep ``max`` behaviour on easy buckets while recovering the
  201+ bucket where the query class is fragmented into several same-scale
  clusters. See ``library/notes/MCV-method.md`` for derivation and failure
  modes; the MAD multiplier ``k`` defaults to ``mask_iqr_k = 1.5`` so MCV
  introduces zero new hyperparameters.

This module is intentionally side-effect free and pure-NumPy, so the same
function is callable from ``pipeline.py`` and from offline analysis of
saved cluster traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .masks import CandidateMask
from .pipeline import OccamResult


PredStrategy = Literal["total", "max", "mode_cluster_vote", "mcv"]


@dataclass(frozen=True)
class PredictTrace:
    """Diagnostic trace of one prediction (saved into per-image JSON)."""

    pred: int
    strategy: str
    cluster_sizes: tuple[int, ...]
    cluster_log_area: tuple[float, ...]
    anchor_index: int | None
    mode_member_indices: tuple[int, ...]
    sigma_log: float | None
    k: float | None

    def to_dict(self) -> dict:
        return {
            "pred": int(self.pred),
            "strategy": self.strategy,
            "cluster_sizes": list(self.cluster_sizes),
            "cluster_log_area": [round(v, 4) for v in self.cluster_log_area],
            "anchor_index": self.anchor_index,
            "mode_member_indices": list(self.mode_member_indices),
            "sigma_log": round(self.sigma_log, 4) if self.sigma_log is not None else None,
            "k": self.k,
        }


def _bbox_area_ratio(mask: CandidateMask, image_area: int) -> float:
    x0, y0, x1, y1 = mask.bbox
    bbox_area = max(0, (x1 - x0)) * max(0, (y1 - y0))
    if image_area <= 0 or bbox_area <= 0:
        return 0.0
    return float(bbox_area) / float(image_area)


def _cluster_log_area(
    indices: tuple[int, ...],
    masks: list[CandidateMask],
    image_area: int,
) -> float:
    ratios = [_bbox_area_ratio(masks[i], image_area) for i in indices]
    ratios = [r for r in ratios if r > 0.0]
    if not ratios:
        return float("-inf")
    return float(np.log10(np.median(ratios)))


def predict_count(
    result: OccamResult,
    strategy: str,
    *,
    image_shape: tuple[int, int] | None = None,
    k: float = 1.5,
    mcv_min_anchor_size: int = 0,
) -> tuple[int, PredictTrace]:
    """Reduce ``result`` to a single count, returning (pred, trace).

    Parameters
    ----------
    result :
        OccamResult containing ``clusters`` and ``masks``.
    strategy :
        One of ``"total"``, ``"max"``, ``"mode_cluster_vote"`` (alias ``"mcv"``).
    image_shape :
        ``(height, width)`` of the input RGB image; required for MCV so
        bbox area can be normalised. Ignored by ``total`` and ``max``.
    k :
        MAD multiplier for the MCV mode-membership fence. Defaults to 1.5
        (matches ``OccamConfig.mask_iqr_k``). Ignored by ``total`` / ``max``.
    mcv_min_anchor_size :
        MCV-only "guard" threshold. When the largest non-singleton cluster
        (the MCV anchor) has fewer than this many members, MCV is bypassed
        and ``max`` is used instead. Defaults to ``0`` (no guard, original
        MCV behaviour). Diagnosed from FSC-147 val 1/3 trace analysis: MCV
        v1 over-counts on small-count images because mode-membership
        promotes same-scale background clusters into the mode-set; gating
        on anchor size avoids this without losing the dense-image gain.
        Ignored by ``total`` / ``max``.
    """

    sizes = tuple(len(c.indices) for c in result.clusters)

    if strategy == "total":
        pred = int(sum(sizes))
        trace = PredictTrace(
            pred=pred,
            strategy=strategy,
            cluster_sizes=sizes,
            cluster_log_area=(),
            anchor_index=None,
            mode_member_indices=tuple(range(len(sizes))),
            sigma_log=None,
            k=None,
        )
        return pred, trace

    if strategy == "max":
        pred = int(max(sizes)) if sizes else 0
        anchor = int(np.argmax(sizes)) if sizes else None
        trace = PredictTrace(
            pred=pred,
            strategy=strategy,
            cluster_sizes=sizes,
            cluster_log_area=(),
            anchor_index=anchor,
            mode_member_indices=(anchor,) if anchor is not None else (),
            sigma_log=None,
            k=None,
        )
        return pred, trace

    if strategy not in ("mode_cluster_vote", "mcv"):
        raise ValueError(f"Unknown pred strategy: {strategy!r}")

    if image_shape is None:
        raise ValueError("mode_cluster_vote requires image_shape=(H, W).")

    height, width = int(image_shape[0]), int(image_shape[1])
    image_area = max(1, height * width)

    if not sizes:
        return 0, PredictTrace(
            pred=0,
            strategy=strategy,
            cluster_sizes=(),
            cluster_log_area=(),
            anchor_index=None,
            mode_member_indices=(),
            sigma_log=None,
            k=k,
        )

    log_areas = tuple(
        _cluster_log_area(c.indices, result.masks, image_area)
        for c in result.clusters
    )

    non_singleton = [i for i, s in enumerate(sizes) if s >= 2 and np.isfinite(log_areas[i])]

    if not non_singleton:
        # Fallback to 'max' if no non-singleton clusters are present.
        anchor = int(np.argmax(sizes))
        pred = int(sizes[anchor])
        trace = PredictTrace(
            pred=pred,
            strategy=strategy + "->max(fallback)",
            cluster_sizes=sizes,
            cluster_log_area=log_areas,
            anchor_index=anchor,
            mode_member_indices=(anchor,),
            sigma_log=None,
            k=k,
        )
        return pred, trace

    sub_sizes = np.array([sizes[i] for i in non_singleton], dtype=int)
    sub_logs = np.array([log_areas[i] for i in non_singleton], dtype=float)
    anchor_local = int(np.argmax(sub_sizes))
    anchor = int(non_singleton[anchor_local])
    u_star = float(sub_logs[anchor_local])

    if mcv_min_anchor_size > 0 and int(sub_sizes[anchor_local]) < mcv_min_anchor_size:
        pred = int(sub_sizes[anchor_local])
        trace = PredictTrace(
            pred=pred,
            strategy=strategy + "->max(guard)",
            cluster_sizes=sizes,
            cluster_log_area=log_areas,
            anchor_index=anchor,
            mode_member_indices=(anchor,),
            sigma_log=None,
            k=k,
        )
        return pred, trace

    deviations = np.abs(sub_logs - u_star)
    sigma_star = float(np.median(deviations))

    if sigma_star == 0.0:
        # Degenerate spread: only one distinct log-area among non-singletons.
        # All matching clusters are considered part of the mode.
        members_local = np.where(deviations == 0.0)[0]
    else:
        fence = k * sigma_star
        members_local = np.where(deviations <= fence)[0]

    members = tuple(int(non_singleton[i]) for i in members_local.tolist())
    pred = int(sum(sizes[i] for i in members)) if members else int(sizes[anchor])

    trace = PredictTrace(
        pred=pred,
        strategy=strategy,
        cluster_sizes=sizes,
        cluster_log_area=log_areas,
        anchor_index=anchor,
        mode_member_indices=members,
        sigma_log=sigma_star,
        k=k,
    )
    return pred, trace


__all__ = ["predict_count", "PredictTrace", "PredStrategy"]
