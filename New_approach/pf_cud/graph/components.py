"""Connected components -> initial counting groups."""

from typing import List

import numpy as np
from scipy.sparse.csgraph import connected_components

from pf_cud.data import CountGroup


def graph_to_groups(graph: np.ndarray) -> List[CountGroup]:
    n = graph.shape[0]
    if n == 0:
        return []

    if n == 1:
        return [CountGroup(indices=[0], count=1)]

    adjacency = (graph > 0).astype(np.int32)

    n_components, labels = connected_components(adjacency, directed=False)

    groups: List[CountGroup] = []
    for c in range(n_components):
        inds = np.where(labels == c)[0].tolist()
        groups.append(CountGroup(indices=inds, count=len(inds)))

    return groups
