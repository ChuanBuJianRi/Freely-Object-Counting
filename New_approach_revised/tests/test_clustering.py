"""Unit tests for affinity + category-aware first-neighbor clustering (numpy)."""

import unittest

import numpy as np

import _pathsetup  # noqa: F401

from ov_cud.clustering.connected_components import connected_components
from ov_cud.clustering.first_neighbor import (
    category_aware_clustering,
    first_neighbor_clustering,
)
from ov_cud.config import ClusterConfig
from ov_cud.matrix.affinity import build_group_affinity


class TestAffinity(unittest.TestCase):
    def test_group_affinity_zero_diagonal(self):
        probs = np.eye(3, dtype=np.float32)
        a_sem = np.ones((3, 3), dtype=np.float32)
        a = build_group_affinity(probs, a_sem)
        self.assertTrue(np.allclose(np.diag(a), 0.0))

    def test_compat_gates_cross_class(self):
        # two classes; cross-class compat is 0 -> affinity 0 even if a_sem high
        probs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        a_sem = np.ones((3, 3), dtype=np.float32)
        a = build_group_affinity(probs, a_sem)
        self.assertAlmostEqual(a[0, 1], 1.0, places=5)   # same class
        self.assertAlmostEqual(a[0, 2], 0.0, places=5)   # different class


class TestConnectedComponents(unittest.TestCase):
    def test_components(self):
        adj = np.zeros((4, 4), dtype=bool)
        adj[0, 1] = adj[1, 0] = True
        adj[2, 3] = adj[3, 2] = True
        comps = connected_components(adj)
        comps = sorted([sorted(c) for c in comps])
        self.assertEqual(comps, [[0, 1], [2, 3]])


class TestFirstNeighbor(unittest.TestCase):
    def test_threshold_isolates(self):
        a = np.array([[0.0, 0.1], [0.1, 0.0]], dtype=np.float32)
        comps = first_neighbor_clustering(a, tau_affinity=0.3)
        self.assertEqual(sorted([sorted(c) for c in comps]), [[0], [1]])
        comps2 = first_neighbor_clustering(a, tau_affinity=0.05)
        self.assertEqual(sorted([sorted(c) for c in comps2]), [[0, 1]])


class TestCategoryAwareClustering(unittest.TestCase):
    def test_groups_by_class_and_partitions(self):
        # 4 candidates: 0,1 -> class A ; 2,3 -> class B
        probs = np.array([
            [0.9, 0.1], [0.85, 0.15], [0.1, 0.9], [0.2, 0.8],
        ], dtype=np.float32)
        a_sem = np.array([
            [0.0, 0.9, 0.8, 0.8],
            [0.9, 0.0, 0.8, 0.8],
            [0.8, 0.8, 0.0, 0.9],
            [0.8, 0.8, 0.9, 0.0],
        ], dtype=np.float32)
        a_group = build_group_affinity(probs, a_sem)
        cfg = ClusterConfig(tau_affinity=0.2, bucket_top_k=1)
        groups = category_aware_clustering(probs, a_group, cfg)

        # every candidate assigned exactly once (partition)
        flat = sorted(i for g in groups for i in g)
        self.assertEqual(flat, [0, 1, 2, 3])
        # 0 and 1 land together; 2 and 3 land together; A-group != B-group
        gid = {}
        for k, g in enumerate(groups):
            for i in g:
                gid[i] = k
        self.assertEqual(gid[0], gid[1])
        self.assertEqual(gid[2], gid[3])
        self.assertNotEqual(gid[0], gid[2])


if __name__ == "__main__":
    unittest.main()
