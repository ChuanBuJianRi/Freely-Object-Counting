"""Configuration for OV-CUD.

Unlike PF-CUD (which was deliberately parameter-free), OV-CUD is allowed tunable
thresholds. They all live here with sensible defaults so the rest of the code is
free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


def default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class FilterConfig:
    """Stage 1 candidate filtering (design.md sec 6.3)."""

    min_area_ratio: float = 1e-4   # mask area / image area lower bound
    max_area_ratio: float = 0.6    # mask area / image area upper bound
    min_source_score: float = 0.0  # SAM stability/iou score lower bound
    dedup_iou: float = 0.9         # drop near-duplicate masks above this IoU
    max_candidates: Optional[int] = 256  # hard cap (keep highest source_score)


@dataclass
class ClusterConfig:
    """Stage 6 clustering (design.md sec 11)."""

    tau_affinity: float = 0.3   # first-neighbor edge threshold on A_group
    bucket_top_k: int = 1       # bucket by top-1 class (k>1 allows multi-bucket)
    min_top_score: float = 0.0  # candidates below this top-class score -> unknown


@dataclass
class HeadConfig:
    category_temperature: float = 0.07  # CLIP-style logit temperature
    label_confidence_floor: float = 0.0  # below -> unknown handling (later stage)


@dataclass
class MatchConfig:
    """Training-time candidate<->GT matching (design.md sec 10.2)."""

    tau_purity: float = 0.5  # below -> background/noise/ignore
    tau_part: float = 0.5    # coverage below + high purity -> part candidate


@dataclass
class Config:
    device: Optional[str] = None

    # Backbone identifiers (only used when real backends are available).
    dinov2_model_name: str = "dinov2_vits14"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    sam2_config: Optional[str] = None
    sam2_checkpoint: Optional[str] = None

    # Force deterministic offline backends (no network / no heavy weights).
    offline: bool = False

    # Crop building. v1 only uses the box crop for both DINOv2 and CLIP.
    crop_size: int = 224
    context_expand: float = 0.25  # used only if context crop is enabled later
    use_masked_crop: bool = False
    use_context_crop: bool = False

    filter: FilterConfig = field(default_factory=FilterConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    match: MatchConfig = field(default_factory=MatchConfig)

    def resolve_device(self) -> str:
        return self.device or default_device()
