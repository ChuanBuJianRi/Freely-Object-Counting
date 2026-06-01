"""Configuration for the OCCAM reimplementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OccamMode = Literal["single", "multi"]


@dataclass(frozen=True)
class OccamConfig:
    """Runtime knobs matching the values reported in the OCCAM paper."""

    mode: OccamMode = "single"
    # ---- SAM2 AMG parameters ----
    amg_points_per_side: int = 32
    amg_pred_iou_thresh: float = 0.7
    amg_stability_score_thresh: float = 0.92
    amg_box_nms_thresh: float = 0.7
    amg_min_mask_region_area: int = 100
    amg_crop_n_layers: int = 1
    # ---- mask filtering ----
    max_mask_area_ratio: float = 0.5
    min_mask_area_ratio: float = 0.0005
    # Filtering policy: p0 area_window | p1 score_thresh | p2 topk |
    # p3 area_iqr | p4 area_otsu | p5 score_and_area | p6 score_nms | p7 no_filter
    mask_policy: str = "p0"
    mask_score_thresh: float = 0.85
    mask_topk: int = 100
    mask_iqr_k: float = 1.5
    # ---- legacy dense-grid fallback (used when AMG not available) ----
    seed_spacing: int = 10
    duplicate_iou_threshold: float = 0.5
    keep_top_k_per_prompt: int = 1
    # ---- multiscale ----
    min_masks_before_multiscale: int = 3
    enable_multiscale: bool = False
    max_multiscale_depth: int = 1
    device: str = "cuda"
    # ---- clustering method ----
    cluster_method: str = "finch"  # "finch" or "sng"
    sng_epsilon: int = 10
    sng_delta: int | None = None  # None ⇒ adaptive via §7.1 (see clustering.adaptive_delta)
    sng_alpha: float = 0.4         # blend coefficient when sng_delta is None
    # ---- prediction strategy ----
    # "total" :: sum of every cluster size (paper default; over-counts noise).
    # "max"   :: size of the largest cluster (FSC-147 default; under-counts
    #            when the query class is fragmented into multiple same-scale
    #            clusters; this is the MP7 best baseline's reduction).
    # "mode_cluster_vote" / "mcv" :: anchored at the largest cluster, sum
    #            every cluster whose log10-area-ratio is within
    #            mask_iqr_k * MAD of the anchor's log-area. Falls back to
    #            "max" when no non-singleton clusters exist. See
    #            library/notes/MCV-method.md for derivation. Reuses the
    #            existing mask_iqr_k value as the MAD multiplier so MCV
    #            adds zero new hyperparameters.
    pred_strategy: str = "total"

    @property
    def crop_size(self) -> int:
        """Bounding box resize size used before ResNet-50 feature extraction."""

        return 224 if self.mode == "single" else 500

    @property
    def finch_thresholds(self) -> tuple[float, ...]:
        """Threshold schedule for the first FINCH iterations."""

        if self.mode == "single":
            return (12.0, 9.0, 7.75)
        return (5.0, 4.0, 3.0)

    @property
    def steady_threshold(self) -> float:
        """Threshold used after the scheduled FINCH iterations."""

        return self.finch_thresholds[-1]

    @classmethod
    def for_mode(
        cls,
        mode: OccamMode,
        *,
        device: str = "cuda",
        enable_multiscale: bool = False,
        min_mask_area_ratio: float | None = None,
        max_mask_area_ratio: float | None = None,
        cluster_method: str | None = None,
        sng_epsilon: int | None = None,
        sng_delta: int | None = None,
        sng_alpha: float | None = None,
        pred_strategy: str | None = None,
        mask_policy: str | None = None,
        mask_score_thresh: float | None = None,
        mask_topk: int | None = None,
        mask_iqr_k: float | None = None,
        seed_spacing: int | None = None,
        duplicate_iou_threshold: float | None = None,
    ) -> "OccamConfig":
        kwargs: dict = dict(mode=mode, device=device, enable_multiscale=enable_multiscale)
        if min_mask_area_ratio is not None:
            kwargs["min_mask_area_ratio"] = min_mask_area_ratio
        if max_mask_area_ratio is not None:
            kwargs["max_mask_area_ratio"] = max_mask_area_ratio
        if cluster_method is not None:
            kwargs["cluster_method"] = cluster_method
        if sng_epsilon is not None:
            kwargs["sng_epsilon"] = sng_epsilon
        if sng_delta is not None:
            kwargs["sng_delta"] = sng_delta
        if sng_alpha is not None:
            kwargs["sng_alpha"] = sng_alpha
        if pred_strategy is not None:
            kwargs["pred_strategy"] = pred_strategy
        if mask_policy is not None:
            kwargs["mask_policy"] = mask_policy
        if mask_score_thresh is not None:
            kwargs["mask_score_thresh"] = mask_score_thresh
        if mask_topk is not None:
            kwargs["mask_topk"] = mask_topk
        if mask_iqr_k is not None:
            kwargs["mask_iqr_k"] = mask_iqr_k
        if seed_spacing is not None:
            kwargs["seed_spacing"] = seed_spacing
        if duplicate_iou_threshold is not None:
            kwargs["duplicate_iou_threshold"] = duplicate_iou_threshold
        return cls(**kwargs)
