"""Scale-layer counting: a parameter-free count read off the blob scale-space.

Why
---
PF-CUD's group-selection ceiling is high because the correct count rarely equals
any single output group's size: over-complete candidates merge across scales
into giant groups, and the right answer is spread across scales. Probing the raw
blob scale-space (the multi-scale LoG that ``BlobCandidateGenerator`` already
computes) shows the count information is cleanly organised *by scale*: each LoG
sigma level holds a number of detections, and one coarse level's detection count
is close to GT (its oracle-scale MAE is far below the group-selection oracle).

Method (parameter-free)
-----------------------
1. Bin blob candidates by their LoG sigma (the generator stores it in
   ``meta['sigma']``); each level has a detection count.
2. The level at which the count matches the true object count is regime
   dependent (coarse for few large objects, fine for many small ones), so a
   single global stability point is biased. We instead pick the *coarsest scale
   still on a count plateau*: levels whose adjacent relative count change is at
   or below the curve's own median relative change are "plateau" levels, and we
   take the coarsest of them. The median split is data-driven (no tuned
   threshold) and adapts the chosen scale per image.
3. The count is that level's number of detections.

Only blob candidates carry a scale; edge/SAM candidates are ignored here. If no
scaled candidate exists the function returns 0. No epsilon / delta / k / size or
response threshold is introduced -- the level is chosen purely from the shape of
the per-scale count curve.
"""

from typing import List

import numpy as np

from pf_cud.data import Candidate


def _per_scale_counts(candidates: List[Candidate]):
    """Return (sorted unique sigmas, detection count per sigma) for blobs."""
    sigmas = [
        float(c.meta["sigma"])
        for c in candidates
        if c.source == "blob" and "sigma" in c.meta
    ]
    return _counts_from_sigmas(sigmas)


def _counts_from_sigmas(sigmas):
    """Bin a flat list of blob sigmas into (levels, per-level counts)."""
    if sigmas is None or len(sigmas) == 0:
        return np.empty(0), np.empty(0)
    s = np.round(np.asarray(sigmas, dtype=np.float64), 4)
    levels = np.array(sorted(set(s.tolist())), dtype=np.float64)
    counts = np.array([int((s == lv).sum()) for lv in levels], dtype=np.float64)
    return levels, counts


def _most_stable_index(counts: np.ndarray) -> int:
    """Index of the scale whose count varies least vs its scale neighbours.

    The per-scale count curve decays then plateaus toward coarse scales; the
    plateau (minimal local variation) marks the object scale. Endpoints are
    excluded since they have only one neighbour. Parameter-free.
    """
    n = len(counts)
    if n == 0:
        return -1
    if n <= 2:
        return n - 1  # coarsest available
    local_var = [
        abs(counts[i - 1] - counts[i]) + abs(counts[i] - counts[i + 1])
        for i in range(1, n - 1)
    ]
    return int(np.argmin(local_var)) + 1


def _coarsest_plateau_index(counts: np.ndarray) -> int:
    """Index of the coarsest scale that still sits on a count plateau.

    Rationale (validated on the full FSC147 test set): the scale at which the
    detection count matches the true object count is *regime dependent* -- it is
    coarse for few large objects and fine for many small ones. ``most_stable``
    picks the globally flattest interior point, which is biased toward the
    coarse tail and overshoots on small-count images.

    Instead we look at the relative change between adjacent levels
    ``|c_{j+1}-c_j| / mean`` and call a level a *plateau* level when its relative
    change is at or below the curve's own median relative change (a data-driven
    split, not a tuned threshold). Among all plateau levels we take the coarsest
    one: it is the last scale before the count starts changing rapidly again,
    i.e. the largest stable object scale supported by the data. This adapts the
    chosen scale per image without any parameter.
    """
    n = len(counts)
    if n == 0:
        return -1
    if n <= 2:
        return n - 1
    rel = np.abs(counts[1:] - counts[:-1]) / (
        0.5 * (counts[1:] + counts[:-1]) + 1.0
    )
    median_rel = float(np.median(rel))
    plateau = np.where(rel <= median_rel)[0]
    if plateau.size == 0:
        return n - 1
    # rel[j] is the change across the (j, j+1) gap; the coarser endpoint of the
    # coarsest low-change gap is the plateau level we keep.
    return int(min(plateau[-1] + 1, n - 1))


def scale_layer_count(candidates: List[Candidate]) -> int:
    """Parameter-free count from the coarsest blob scale plateau (0 if none)."""
    levels, counts = _per_scale_counts(candidates)
    if counts.size == 0:
        return 0
    idx = _coarsest_plateau_index(counts)
    return int(counts[idx])


def scale_layer_count_from_sigmas(sigmas) -> int:
    """Same as ``scale_layer_count`` but from a raw blob-sigma list.

    Use this with the pre-dedup sigmas in ``CountResult.meta['raw_blob_sigmas']``
    so the per-scale curve is not distorted by candidate deduplication.
    """
    levels, counts = _counts_from_sigmas(sigmas)
    if counts.size == 0:
        return 0
    return int(counts[_coarsest_plateau_index(counts)])


def scale_layer_detail(candidates: List[Candidate]) -> dict:
    """Diagnostic view: chosen count, chosen sigma, and the full per-scale curve."""
    levels, counts = _per_scale_counts(candidates)
    if counts.size == 0:
        return {"count": 0, "sigma": None, "levels": [], "counts": []}
    idx = _coarsest_plateau_index(counts)
    return {
        "count": int(counts[idx]),
        "sigma": float(levels[idx]),
        "levels": levels.tolist(),
        "counts": counts.astype(int).tolist(),
    }
