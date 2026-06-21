"""candidates subpackage：候选裁剪与几何特征。"""

from .crops import build_three_crops
from .geometry import containment, geometry_features, mask_iou, mask_overlaps

__all__ = [
    "build_three_crops",
    "geometry_features",
    "mask_overlaps",
    "mask_iou",
    "containment",
]
