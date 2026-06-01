"""Core data structures for PF-CUD."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Candidate:
    """一个候选可数区域。"""

    mask: np.ndarray  # bool array, shape = [H, W]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    source: str  # "sam", "blob", "edge", etc.
    score: Optional[float] = None
    features: Dict[str, np.ndarray] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CountGroup:
    """一个最终或中间的可数单元组。"""

    indices: List[int]  # candidate indices
    group_type: Optional[str] = None  # object / pattern / background / unknown
    count: Optional[int] = None
    confidence: Optional[float] = None
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CountResult:
    """整张图片的输出结果。"""

    groups: List[CountGroup]
    candidates: List[Candidate]
    image_shape: Tuple[int, int]
    meta: Dict[str, Any] = field(default_factory=dict)
