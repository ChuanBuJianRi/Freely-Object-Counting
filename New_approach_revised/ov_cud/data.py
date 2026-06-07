"""Core data structures for OV-CUD (design.md section 16).

The pipeline in this package stops at stage 6 (coarse semantic groups), so the
top-level result holds ``CoarseSemanticGroup`` objects rather than final counted
``SemanticCountGroup`` objects. Refinement (sec 12), instance deduplication /
counting (sec 13) and label aggregation (sec 14) are intentionally out of scope
for this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass
class Candidate:
    """An over-complete mask/box candidate (not necessarily a real instance)."""

    mask: np.ndarray  # bool array [H, W]
    bbox: BBox
    area: float
    source: str = "sam2"
    source_score: Optional[float] = None
    # Per-candidate artifacts filled in as the pipeline runs.
    crops: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    geometry: Optional[np.ndarray] = None
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    predictions: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticInstance:
    """A single representative candidate, surfaced as an instance."""

    candidate_index: int
    bbox: BBox
    class_name: str
    confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoarseSemanticGroup:
    """A coarse (pre-refinement, pre-counting) semantic group from stage 6.

    ``n_candidates`` is the number of candidates in the group, NOT a final count.
    Counting happens in a later milestone (design.md sec 13).
    """

    class_name: str
    candidate_indices: List[int]
    n_candidates: int
    confidence: float
    is_countable: bool
    class_distribution: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticCountResult:
    """Whole-image output (coarse groups for this milestone)."""

    groups: List[CoarseSemanticGroup]
    candidates: List[Candidate]
    image_shape: Tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "image_shape": list(self.image_shape),
            "groups": [
                {
                    "class_name": g.class_name,
                    "n_candidates": g.n_candidates,
                    "confidence": round(float(g.confidence), 4),
                    "is_countable": bool(g.is_countable),
                    "candidate_indices": list(g.candidate_indices),
                    "class_distribution": {
                        k: round(float(v), 4) for k, v in g.class_distribution.items()
                    },
                }
                for g in self.groups
            ],
            "meta": self.meta,
        }
