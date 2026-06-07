"""End-to-end inference smoke test (offline, no torch required).

Exercises stages 1-6 with deterministic offline encoders + the heuristic
relation head, asserting structural correctness and determinism (semantic
quality is not asserted -- offline encoders carry no semantics).
"""

import json
import unittest

import numpy as np

import _pathsetup  # noqa: F401

from ov_cud.config import Config
from ov_cud.data import Candidate
from ov_cud.pipeline import build_default_pipeline
from ov_cud.training.dataset import SyntheticGTDataset


def _make_proposal_fn(sample):
    raw = []
    for inst in sample.instances:
        m = inst.mask.copy()
        raw.append(Candidate(mask=m, bbox=inst.bbox, area=float(m.sum()),
                             source="gt", source_score=1.0))
    # a couple of distractor boxes
    h, w = sample.image.shape[:2]
    for (x1, y1, x2, y2) in [(0, 0, w // 5, h // 5), (w - w // 5, h - h // 5, w, h)]:
        m = np.zeros((h, w), dtype=bool)
        m[y1:y2, x1:x2] = True
        raw.append(Candidate(mask=m, bbox=(x1, y1, x2, y2), area=float(m.sum()),
                             source="distractor", source_score=0.6))

    def proposal_fn(_image):
        return [Candidate(mask=c.mask.copy(), bbox=c.bbox, area=c.area,
                          source=c.source, source_score=c.source_score) for c in raw]

    return proposal_fn


class TestPipelineSmoke(unittest.TestCase):
    def setUp(self):
        self.sample = list(SyntheticGTDataset(n_images=1, image_size=96, seed=3))[0]
        self.config = Config(offline=True)

    def _run(self):
        pipeline = build_default_pipeline(self.config)
        pipeline.proposal_fn = _make_proposal_fn(self.sample)
        return pipeline.run(self.sample.image)

    def test_runs_and_is_wellformed(self):
        result = self._run()
        n = len(result.candidates)
        self.assertGreater(n, 0)

        # every candidate has a category distribution
        for c in result.candidates:
            probs = c.predictions["category_probs"]
            self.assertEqual(probs.ndim, 1)
            self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)
            self.assertIn("top_class", c.predictions)

        # groups partition the candidate set exactly
        flat = sorted(i for g in result.groups for i in g.candidate_indices)
        self.assertEqual(flat, list(range(n)))
        self.assertEqual(result.meta["n_candidates"], n)
        self.assertEqual(result.meta["relation_head"], "HeuristicRelationHead")

    def test_json_serializable(self):
        result = self._run()
        text = json.dumps(result.to_json())
        self.assertIn("groups", json.loads(text))

    def test_deterministic(self):
        a = self._run().to_json()
        b = self._run().to_json()
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
