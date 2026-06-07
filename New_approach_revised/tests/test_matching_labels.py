"""Unit tests for candidate<->GT matching and label generation (numpy only)."""

import unittest

import numpy as np

import _pathsetup  # noqa: F401

from ov_cud.config import MatchConfig
from ov_cud.training.dataset import GTInstance
from ov_cud.training.labels import relation_pair_targets, category_targets
from ov_cud.training.matching import match_candidates_to_gt
from ov_cud.vocabulary import VocabularyBank


def _box_mask(shape, x1, y1, x2, y2):
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def _gt(mask, cls, iid):
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return GTInstance(mask=mask, class_name=cls, instance_id=iid, bbox=bbox)


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.shape = (40, 40)
        self.cfg = MatchConfig(tau_purity=0.5, tau_part=0.5)
        self.gA = _gt(_box_mask(self.shape, 2, 2, 12, 12), "apple", 0)
        self.gB = _gt(_box_mask(self.shape, 20, 20, 34, 34), "car", 1)

    def test_exact_match_is_valid(self):
        cand = _box_mask(self.shape, 2, 2, 12, 12)  # == gA
        r = match_candidates_to_gt([cand], [self.gA, self.gB], self.cfg)
        self.assertAlmostEqual(r.iou[0], 1.0, places=5)
        self.assertAlmostEqual(r.purity[0], 1.0, places=5)
        self.assertAlmostEqual(r.coverage[0], 1.0, places=5)
        self.assertEqual(r.matched_class[0], "apple")
        self.assertEqual(r.matched_instance_id[0], 0)
        self.assertEqual(r.validity[0], "valid")
        self.assertTrue(r.is_valid[0])

    def test_high_purity_low_coverage_is_part(self):
        # candidate is a small sub-region fully inside gA: purity high, coverage low
        cand = _box_mask(self.shape, 3, 3, 6, 6)
        r = match_candidates_to_gt([cand], [self.gA, self.gB], self.cfg)
        self.assertGreaterEqual(r.purity[0], self.cfg.tau_purity)
        self.assertLess(r.coverage[0], self.cfg.tau_part)
        self.assertEqual(r.validity[0], "part")
        self.assertEqual(r.matched_class[0], "apple")

    def test_background_candidate(self):
        cand = _box_mask(self.shape, 36, 0, 40, 4)  # empty corner, no overlap
        r = match_candidates_to_gt([cand], [self.gA, self.gB], self.cfg)
        self.assertEqual(r.validity[0], "background")
        self.assertFalse(r.is_valid[0])
        self.assertEqual(r.matched_class[0], "")
        self.assertEqual(r.weight[0], 0.0)


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.shape = (40, 40)
        self.cfg = MatchConfig(tau_purity=0.5, tau_part=0.5)
        self.vocab = VocabularyBank.for_classes(["apple", "car"])

    def _match(self, cands, gts):
        return match_candidates_to_gt(cands, gts, self.cfg)

    def test_category_targets_ignore_background(self):
        gA = _gt(_box_mask(self.shape, 2, 2, 12, 12), "apple", 0)
        cands = [_box_mask(self.shape, 2, 2, 12, 12), _box_mask(self.shape, 36, 36, 40, 40)]
        r = self._match(cands, [gA])
        idx, w = category_targets(r, self.vocab)
        self.assertEqual(idx[0], self.vocab.index_of("apple"))
        self.assertGreater(w[0], 0.0)
        self.assertEqual(idx[1], -1)   # background -> ignore
        self.assertEqual(w[1], 0.0)

    def test_relation_targets_same_instance_and_class(self):
        gA = _gt(_box_mask(self.shape, 2, 2, 12, 12), "apple", 0)
        gA2 = _gt(_box_mask(self.shape, 20, 2, 30, 12), "apple", 1)
        gB = _gt(_box_mask(self.shape, 20, 20, 34, 34), "car", 2)
        cands = [gA.mask, gA2.mask, gB.mask]
        r = self._match(cands, [gA, gA2, gB])
        containment = np.zeros((3, 3), dtype=np.float32)
        pairs = [(0, 1), (0, 2)]
        t = relation_pair_targets(r, containment, pairs)
        # (apple0, apple1): same class, diff instance
        self.assertEqual(t["y_sem"][0], 1.0)
        self.assertEqual(t["y_inst"][0], 0.0)
        # (apple0, car2): different class
        self.assertEqual(t["y_sem"][1], 0.0)
        self.assertEqual(t["y_inst"][1], 0.0)

    def test_part_target_directional(self):
        whole = _box_mask(self.shape, 2, 2, 22, 22)   # large
        part = _box_mask(self.shape, 3, 3, 8, 8)       # inside whole
        gW = _gt(whole, "apple", 0)
        # both candidates matched to the same instance (the whole apple)
        cands = [part, whole]
        r = self._match(cands, [gW])
        # containment[part, whole] high; completeness_gap(whole>part) positive
        containment = np.array([[0.0, 1.0], [0.25, 0.0]], dtype=np.float32)
        t = relation_pair_targets(r, containment, [(0, 1), (1, 0)])
        self.assertGreater(t["y_part"][0], 0.0)        # part -> whole
        self.assertEqual(t["y_part"][1], 0.0)          # whole -> part is 0


if __name__ == "__main__":
    unittest.main()
