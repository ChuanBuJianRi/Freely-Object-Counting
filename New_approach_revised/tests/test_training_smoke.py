"""Training smoke tests (require torch).

Verifies the Stage 1 (category) and Stage 2 (relation) training loops run on the
synthetic dataset with offline encoders, produce finite & non-increasing loss,
and yield heads that plug into inference. No real dataset / network involved.
"""

import math
import unittest

import numpy as np

import _pathsetup  # noqa: F401

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

from ov_cud.config import Config
from ov_cud.encoders.clip_encoder import build_clip_encoder
from ov_cud.encoders.dinov2_encoder import build_region_encoder
from ov_cud.training.dataset import SyntheticGTDataset
from ov_cud.vocabulary import VocabularyBank


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestTrainingSmoke(unittest.TestCase):
    def setUp(self):
        self.config = Config(offline=True)
        self.ds = SyntheticGTDataset(n_images=6, image_size=96, seed=2)
        self.vocab = VocabularyBank.for_classes(self.ds.class_names)
        self.clip = build_clip_encoder(self.config)
        self.region = build_region_encoder(self.config)

    def test_train_category(self):
        from ov_cud.training.train_category import CategoryTrainConfig, train_category

        head, history = train_category(
            self.ds, self.clip, self.vocab, self.config,
            CategoryTrainConfig(epochs=5, lr=5e-3, seed=0),
        )
        self.assertEqual(len(history), 5)
        self.assertTrue(all(math.isfinite(h) for h in history))
        self.assertLessEqual(history[-1], history[0] + 1e-6)
        self.assertIsNotNone(head.projection)
        self.assertGreater(head.temperature, 0.0)

        probs = head.predict([np.zeros((20, 20, 3), dtype=np.uint8)])
        self.assertEqual(probs.shape, (1, self.vocab.num_classes))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=4)

    def test_train_relation_and_inference(self):
        from ov_cud.heads.relation_head import LearnedRelationHead
        from ov_cud.training.train_relation import RelationTrainConfig, train_relation

        head, history = train_relation(
            self.ds, self.region, self.clip, self.vocab, self.config,
            RelationTrainConfig(epochs=5, lr=5e-3, hidden=64, max_pairs=128, seed=0),
        )
        self.assertEqual(len(history), 5)
        self.assertTrue(all(math.isfinite(h) for h in history))
        self.assertLessEqual(history[-1], history[0] + 1e-6)
        self.assertIsInstance(head, LearnedRelationHead)

        # plug the learned head into a full pipeline run.
        from ov_cud.data import Candidate
        from ov_cud.pipeline import build_default_pipeline

        sample = list(self.ds)[0]

        def proposal_fn(_img):
            out = []
            for inst in sample.instances:
                m = inst.mask.copy()
                out.append(Candidate(mask=m, bbox=inst.bbox, area=float(m.sum()),
                                     source="gt", source_score=1.0))
            return out

        pipeline = build_default_pipeline(self.config, self.vocab)
        pipeline.proposal_fn = proposal_fn
        pipeline.relation_head = head
        result = pipeline.run(sample.image)
        n = len(result.candidates)
        flat = sorted(i for g in result.groups for i in g.candidate_indices)
        self.assertEqual(flat, list(range(n)))
        self.assertEqual(result.meta["relation_head"], "LearnedRelationHead")

    def test_learned_head_matrix_properties(self):
        from ov_cud.heads.category_head import ClipProjectionCategoryHead
        from ov_cud.matrix.pairwise_features import build_pairwise_context
        from ov_cud.training.candidate_source import candidates_from_gt
        from ov_cud.training.train_relation import RelationTrainConfig, train_relation

        head, _ = train_relation(
            self.ds, self.region, self.clip, self.vocab, self.config,
            RelationTrainConfig(epochs=2, hidden=64, max_pairs=64, seed=0),
        )
        sample = list(self.ds)[0]
        cands = candidates_from_gt(sample, self.config, seed=0)
        crops = [c.crops["box"] for c in cands]
        z = self.region.encode(crops)
        cat = ClipProjectionCategoryHead(self.clip, self.vocab)
        cat.build_prototypes()
        probs = cat.predict(crops)
        ctx = build_pairwise_context(cands, z, probs, sample.image.shape)
        rel = head(ctx)
        n = ctx.n
        for key in ("A_sem", "A_inst", "A_part"):
            self.assertEqual(rel[key].shape, (n, n))
            self.assertTrue(np.all(rel[key] >= -1e-6) and np.all(rel[key] <= 1 + 1e-6))
        # sem/inst symmetric, diagonal zero
        self.assertTrue(np.allclose(rel["A_sem"], rel["A_sem"].T, atol=1e-5))
        self.assertTrue(np.allclose(np.diag(rel["A_inst"]), 0.0))


if __name__ == "__main__":
    unittest.main()
