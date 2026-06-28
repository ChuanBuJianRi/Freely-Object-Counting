"""candidates subpackage：候选裁剪、几何特征、过滤与候选-GT 匹配。"""

from .crops import build_three_crops
from .filtering import filter_candidates
from .geometry import containment, geometry_features, mask_iou, mask_overlaps
from .matching import (
    GTInstance,
    MatchResult,
    match_candidate,
    match_candidates,
    stack_match_labels,
)

__all__ = [
    "build_three_crops",
    "filter_candidates",
    "geometry_features",
    "mask_overlaps",
    "mask_iou",
    "containment",
    "GTInstance",
    "MatchResult",
    "match_candidate",
    "match_candidates",
    "stack_match_labels",
]
