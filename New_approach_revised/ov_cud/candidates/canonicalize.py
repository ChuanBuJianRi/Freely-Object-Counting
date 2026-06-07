"""Attach crops + geometry to candidates (design.md sec 6.2)."""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import Config
from ..data import Candidate
from .crops import build_crops
from .geometry import compute_geometry


def canonicalize(
    image: np.ndarray, candidates: List[Candidate], config: Config
) -> List[Candidate]:
    for c in candidates:
        c.crops = build_crops(
            image,
            c.mask,
            c.bbox,
            use_masked=config.use_masked_crop,
            use_context=config.use_context_crop,
            context_expand=config.context_expand,
        )
        c.geometry = compute_geometry(c.mask, c.bbox, image.shape)
    return candidates
