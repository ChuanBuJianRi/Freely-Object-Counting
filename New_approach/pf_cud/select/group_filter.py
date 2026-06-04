"""Adaptive, parameter-free selection of the counting groups.

Problem
-------
PF-CUD's grouping stage recovers good *units* (the oracle group is usually close
to GT), but neither extreme of the group-size spectrum is the right answer:

* "take the rank-1 group" collapses to a tiny, over-pure fragment (count ~ 2-3);
* "take the biggest / most-populous group" lands on a giant blob that merged
  many overlapping over-complete candidates of different scales/objects.

The true countable unit sits in between: it is the group whose members look most
like *the same template instantiated many times*. Giant noise groups lose
because their members are internally inconsistent; tiny fragments lose because
they barely repeat.

Scoring
-------
For every kept group we combine two data-driven, weight-free cues:

* repetition scale = log(count) -- rewards "appears many times" but the log
  damps runaway giant groups so raw size cannot dominate;
* internal consistency = 1 / (mean pairwise fused feature distance) -- high when
  members are near-identical (a replicated template), low for the heterogeneous
  mega-groups produced by over-complete candidate merging.

Both cues are rank-normalised and averaged (no manual weights, mirroring the
project's rank-aggregation style). The group with the best fused rank is the
dominant counting unit.

Keep-set selection (which groups are countable foreground at all) drops
``background_or_pattern`` groups and Otsu-splits the consistency-weighted
repetition signal, keeping the high cluster.

Everything is parameter-free: only Otsu, rank normalisation, log and the
existing fused-distance are used -- no epsilon / delta / k / size threshold.
"""

from typing import List

import numpy as np

from pf_cud.data import Candidate, CountGroup
from pf_cud.features.fusion import fused_distance

try:
    from skimage.filters import threshold_otsu
except Exception:  # pragma: no cover
    threshold_otsu = None

_BACKGROUND_TYPE = "background_or_pattern"


def group_total_area_frac(
    candidates: List[Candidate], group: CountGroup, image_area: float
) -> float:
    """Sum of member mask areas divided by the image area (total coverage)."""
    if image_area <= 0:
        return 0.0
    total = sum(int(candidates[i].mask.sum()) for i in group.indices)
    return float(total) / float(image_area)


def group_internal_consistency(
    candidates: List[Candidate], group: CountGroup
) -> float:
    """How template-like the members are: 1 / (mean pairwise fused distance).

    A high value means the members are near-identical repeated instances (a real
    countable class); a low value flags the heterogeneous mega-groups that
    over-complete candidate merging produces. Singletons/pairs with no spread
    return a neutral 0 so they neither win nor poison the ranking.
    """
    idx = group.indices
    if len(idx) <= 1:
        return 0.0
    members = [candidates[i] for i in idx]
    d = fused_distance(members)  # symmetric (m, m), zero diagonal
    m = d.shape[0]
    iu = np.triu_indices(m, k=1)
    vals = d[iu]
    if vals.size == 0:
        return 0.0
    mean_d = float(vals.mean())
    if mean_d <= 0.0:
        # identical members -> maximally consistent.
        return float("inf")
    return 1.0 / mean_d


def _rank01(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    """Map values to [0, 1] ranks (1 = best), robust to outliers and +/-inf.

    Mirrors ``features.fusion.rank_normalize_distance`` /
    ``ranking.rank_values``: rank normalisation rather than min-max, so a single
    giant value cannot distort the scale.
    """
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.ones(1, dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    hi = finite.max() if finite.size else 0.0
    lo = finite.min() if finite.size else 0.0
    arr = np.where(np.isposinf(arr), hi, arr)
    arr = np.where(np.isneginf(arr), lo, arr)

    order = np.argsort(arr)
    if higher_is_better:
        order = order[::-1]
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 1.0 - ranks / (n - 1)


def _unit_strength(
    candidates: List[Candidate], groups: List[CountGroup]
) -> np.ndarray:
    """Countable-unit strength = mean rank of (repetition scale, consistency).

    repetition scale  = log1p(count)         -- damped so giant groups can't win
                                                on size alone.
    internal consistency = 1 / mean pairwise fused distance -- template-likeness.
    """
    rep = np.array([np.log1p(len(g.indices)) for g in groups], dtype=np.float64)
    cons = np.array(
        [group_internal_consistency(candidates, g) for g in groups],
        dtype=np.float64,
    )
    rep_rank = _rank01(rep, higher_is_better=True)
    cons_rank = _rank01(cons, higher_is_better=True)
    return 0.5 * (rep_rank + cons_rank)


def select_counting_groups(
    candidates: List[Candidate],
    groups: List[CountGroup],
    image_shape,
) -> List[CountGroup]:
    """Return the groups worth counting, ranked, dominant unit first.

    Stage 1 (keep-set): drop ``background_or_pattern`` groups, then Otsu-split
    the unit-strength signal and keep the high cluster.
    Stage 2 (rank): order survivors by the same unit-strength score.
    Parameter-free.
    """
    if not groups:
        return []

    h, w = image_shape[:2]
    image_area = float(h * w)

    foreground = [g for g in groups if g.group_type != _BACKGROUND_TYPE]
    if not foreground:
        foreground = list(groups)

    if len(foreground) == 1:
        kept = foreground
    else:
        strength = _unit_strength(candidates, foreground)
        uniq = np.unique(strength)
        if uniq.size <= 1 or threshold_otsu is None:
            kept = foreground
        else:
            tau = threshold_otsu(strength)
            kept = [g for g, s in zip(foreground, strength) if s >= tau]
            if not kept:  # degenerate Otsu cut -> keep all foreground
                kept = foreground

    # Stage 2: rank survivors by unit strength (recomputed over the keep-set so
    # ranks reflect the surviving distribution).
    final = _unit_strength(candidates, kept)

    for g, s in zip(kept, final):
        a = group_total_area_frac(candidates, g, image_area)
        g.count = len(g.indices)
        g.score = float(s)
        g.confidence = float(s)
        g.meta["select_score"] = float(s)
        g.meta["select_area_frac"] = float(a)
        g.group_type = "object_or_counting_unit"

    order = np.argsort(final)[::-1]
    return [kept[i] for i in order]


def select_count(
    candidates: List[Candidate],
    groups: List[CountGroup],
    image_shape,
) -> int:
    """Convenience: count of the top-ranked counting group (0 if none)."""
    ranked = select_counting_groups(candidates, groups, image_shape)
    return len(ranked[0].indices) if ranked else 0
