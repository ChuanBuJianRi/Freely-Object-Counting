"""encoders subpackage：冻结骨干的轻量封装。"""

from .dinov2_encoder import DINOv2RegionEncoder
from .text_encoder import TextPrototypeBuilder

__all__ = ["DINOv2RegionEncoder", "TextPrototypeBuilder"]
