"""Turn GT instances into training candidates.

For a real run this is replaced by SAM2 proposals (matched to GT). The default
here derives candidates from GT masks (optionally jittered, plus a few random
background boxes) so the training pipeline runs with no SAM2 dependency. Crops
and geometry are attached so encoders and pairwise features work unchanged.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..candidates.canonicalize import canonicalize
from ..config import Config
from ..data import Candidate
from .dataset import ImageSample


def _bbox_of(mask: np.ndarray):
    ys, xs = np.where(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def candidates_from_gt(
    sample: ImageSample,
    config: Config,
    n_distractors: int = 2,
    seed: int = 0,
) -> List[Candidate]:
    rng = np.random.default_rng(seed)
    h, w = sample.image.shape[:2]
    cands: List[Candidate] = []

    for inst in sample.instances:
        mask = inst.mask.copy()
        cands.append(Candidate(mask=mask, bbox=_bbox_of(mask), area=float(mask.sum()),
                               source="gt", source_score=1.0))

    # random background boxes -> low-purity candidates (background/ignore labels)
    for _ in range(n_distractors):
        bw = int(rng.integers(max(2, w // 10), max(3, w // 4)))
        bh = int(rng.integers(max(2, h // 10), max(3, h // 4)))
        x1 = int(rng.integers(0, max(1, w - bw)))
        y1 = int(rng.integers(0, max(1, h - bh)))
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y1 + bh, x1:x1 + bw] = True
        cands.append(Candidate(mask=mask, bbox=(x1, y1, x1 + bw, y1 + bh),
                               area=float(mask.sum()), source="distractor",
                               source_score=0.5))

    canonicalize(sample.image, cands, config)
    return cands
