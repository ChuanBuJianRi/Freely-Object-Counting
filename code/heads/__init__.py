"""heads subpackage."""

from .category_head import (
    AuxiliaryHeads,
    CategoryHeadConfig,
    CosineMarginHead,
    HybridCategoryHead,
    LinearPrototypeHead,
    ProjectionHead,
    TextPrototypeCosineHead,
    build_category_head,
)

__all__ = [
    "AuxiliaryHeads",
    "CategoryHeadConfig",
    "CosineMarginHead",
    "HybridCategoryHead",
    "LinearPrototypeHead",
    "ProjectionHead",
    "TextPrototypeCosineHead",
    "build_category_head",
]
