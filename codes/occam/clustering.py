"""Thresholded FINCH-style clustering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cluster:
    """Indices of candidate masks assigned to one discovered object class."""

    indices: tuple[int, ...]


def thresholded_finch(
    features: np.ndarray,
    *,
    thresholds: tuple[float, ...],
    steady_threshold: float,
) -> list[Cluster]:
    """Cluster features using the OCCAM paper's thresholded FINCH variant.

    This is a practical reconstruction from the method description. Each
    iteration computes cluster centroids, finds first-neighbor relations, makes
    the relation symmetric, and merges connected components only when centroid
    distance is under the active threshold.
    """

    if len(features) == 0:
        return []
    if len(features) == 1:
        return [Cluster((0,))]

    clusters: list[tuple[int, ...]] = [(index,) for index in range(len(features))]
    iteration = 0

    while True:
        threshold = thresholds[iteration] if iteration < len(thresholds) else steady_threshold
        centroids = np.stack([features[list(cluster)].mean(axis=0) for cluster in clusters])
        nearest = _nearest_neighbors(centroids)

        parent = list(range(len(clusters)))

        for source, target in enumerate(nearest):
            distance = np.linalg.norm(centroids[source] - centroids[target])
            if distance <= threshold:
                _union(parent, source, target)

        merged = _materialize_components(parent, clusters)
        if merged == clusters:
            break

        clusters = merged
        iteration += 1
        if len(clusters) == 1:
            break

    return [Cluster(tuple(cluster)) for cluster in clusters]


def sng_cluster(
    features: np.ndarray,
    *,
    epsilon: int,
    delta: int | None = None,
    alpha: float = 0.4,
) -> list[Cluster]:
    """Epsilon-Delta Shared-Neighbor Graph clustering.

    Each node connects to its ``epsilon`` nearest neighbours (undirected: an
    edge A-B exists iff A is in epsilon-NN(B) OR B is in epsilon-NN(A)). Then
    every edge whose endpoints share at most ``delta`` common neighbours in
    the current graph is removed. Final clusters are the connected components
    of the surviving graph.

    When ``delta`` is None the threshold is auto-derived from ``(epsilon, n)``
    via the §7.1 formula in ``library/notes/SNG-method.md``::

        delta* = floor(alpha * (epsilon - 1) + (1 - alpha) * epsilon**2 / n)

    which pins the dimensionless health-index ``eta`` at ``alpha``. Sweet
    spot is ``alpha in [0.3, 0.5]``; default 0.4 matches the empirical
    sweet spot ``eta ≈ 0.4`` (see results/2026-05-20-...-clustering-sng).
    Pass an integer ``delta`` to bypass the adaptive rule (legacy behaviour).
    """

    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [Cluster((0,))]

    if delta is None:
        delta = adaptive_delta(epsilon=epsilon, n=n, alpha=alpha)

    distances = _pairwise_distances(features)
    np.fill_diagonal(distances, np.inf)
    eps = max(1, min(int(epsilon), n - 1))
    knn = np.argpartition(distances, eps - 1, axis=1)[:, :eps]

    adjacency: list[set[int]] = [set() for _ in range(n)]
    for source in range(n):
        for target in knn[source]:
            t = int(target)
            if t == source:
                continue
            adjacency[source].add(t)
            adjacency[t].add(source)

    edges = {
        (a, b) for a in range(n) for b in adjacency[a] if a < b
    }
    surviving: list[set[int]] = [set() for _ in range(n)]
    for a, b in edges:
        common = len(adjacency[a] & adjacency[b])
        if common > delta:
            surviving[a].add(b)
            surviving[b].add(a)

    parent = list(range(n))
    for a in range(n):
        for b in surviving[a]:
            if a < b:
                _union(parent, a, b)

    components: dict[int, list[int]] = {}
    for index in range(n):
        components.setdefault(_find(parent, index), []).append(index)

    return [Cluster(tuple(sorted(indices))) for indices in components.values()]


def _nearest_neighbors(centroids: np.ndarray) -> np.ndarray:
    distances = _pairwise_distances(centroids)
    np.fill_diagonal(distances, np.inf)
    return distances.argmin(axis=1)


def adaptive_delta(*, epsilon: int, n: int, alpha: float = 0.4) -> int:
    """Auto-derive the SNG shared-neighbour threshold delta from (epsilon, n).

    Formula (§7.1 of library/notes/SNG-method.md)::

        delta* = floor(alpha * (epsilon - 1) + (1 - alpha) * epsilon**2 / n)

    The two terms bracket delta from below (zero-model noise expectation
    epsilon**2/n) and above (intra-class shared-neighbour upper bound
    epsilon - 1). alpha = 0 reduces to the noise floor (no pruning),
    alpha = 1 saturates at the upper bound (over-pruning). Empirically
    alpha in [0.3, 0.5] keeps the health index eta near 0.4 — the
    sweet spot observed on FSC-147 (run 2026-05-20-0942-ablation-clustering-sng).

    The returned value is clamped to ``[0, max(0, epsilon - 2)]`` so that:
    - delta >= 0 is always a meaningful integer threshold;
    - delta <= epsilon - 2 leaves room for at least one intra-class edge to
      survive even at small n (the strict §6.2 upper bound is ``delta < epsilon - 1``).
    """
    if epsilon < 1:
        raise ValueError(f"epsilon must be >= 1, got {epsilon}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    raw = alpha * (epsilon - 1) + (1.0 - alpha) * (epsilon ** 2) / n
    upper = max(0, epsilon - 2)
    return int(max(0, min(upper, np.floor(raw))))


def eta_health(*, epsilon: int, delta: int, n: int) -> float:
    """Dimensionless SNG health index eta (§6.4 of SNG-method.md).

        eta = (delta - epsilon**2/n) / (epsilon - 1 - epsilon**2/n)

    eta in (0, 1) ⇒ feasible region; eta ≈ 0.4–0.5 is the sweet spot;
    eta < 0 ⇒ noise dominates (degenerates to plain epsilon-NN);
    eta > 1 ⇒ delta exceeds the intra-class upper bound (cluster collapse).

    Returns NaN when the denominator vanishes (epsilon - 1 == epsilon**2/n,
    i.e. the existence condition for a feasible delta is just barely met).
    """
    noise = (epsilon ** 2) / n
    denom = (epsilon - 1) - noise
    if denom == 0:
        return float("nan")
    return float((delta - noise) / denom)


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(values * values, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * values @ values.T
    return np.sqrt(np.maximum(distances, 0.0))


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def _union(parent: list[int], first: int, second: int) -> None:
    root_first = _find(parent, first)
    root_second = _find(parent, second)
    if root_first != root_second:
        parent[root_second] = root_first


def _materialize_components(
    parent: list[int],
    clusters: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    components: dict[int, list[int]] = {}
    for cluster_index, cluster in enumerate(clusters):
        root = _find(parent, cluster_index)
        components.setdefault(root, []).extend(cluster)

    return [tuple(sorted(indices)) for indices in components.values()]
