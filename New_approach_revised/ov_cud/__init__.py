"""OV-CUD: Open-Vocabulary Counting Unit Discovery.

Inference skeleton for pipeline stages 1-6 (SAM2 proposal -> canonicalization ->
DINOv2 region encoding -> CLIP category head -> pairwise relation head ->
category-aware clustering), plus training tracks for the category projection
head (Stage 1) and the relation head (Stage 2).

Heavy backends (SAM2, DINOv2 hub weights, CLIP / open_clip) are injectable and
have deterministic offline fallbacks so the full pipeline and its smoke tests
run with no network access and no extra heavy dependencies.
"""

from .data import (
    Candidate,
    CoarseSemanticGroup,
    SemanticCountResult,
    SemanticInstance,
)

__all__ = [
    "Candidate",
    "CoarseSemanticGroup",
    "SemanticCountResult",
    "SemanticInstance",
]
