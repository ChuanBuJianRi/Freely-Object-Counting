"""Stage 2 training: pairwise relation head (design.md sec 10.4, 10.5).

Frozen SAM2 / DINOv2 / CLIP / category head. Trainable: an MLP over phi_ij.
Loss = weighted BCE on (same-class, same-instance, part-whole) targets over
sampled pairs. Runs on SyntheticGTDataset with offline encoders, so it is
testable without a real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..heads.category_head import ClipProjectionCategoryHead
from ..heads.relation_head import LearnedRelationHead, build_relation_mlp
from ..matrix.pairwise_features import build_pairwise_context, build_phi_pairs
from ..vocabulary import VocabularyBank
from .candidate_source import candidates_from_gt
from .dataset import GTInstanceDataset
from .labels import relation_pair_targets
from .losses import weighted_bce_with_logits
from .matching import match_candidates_to_gt
from .sampling import sample_pairs


@dataclass
class RelationTrainConfig:
    epochs: int = 4
    lr: float = 1e-3
    hidden: int = 128
    max_pairs: int = 256
    seed: int = 0


def train_relation(
    dataset: GTInstanceDataset,
    region_encoder,
    clip_encoder,
    vocabulary: VocabularyBank,
    config: Optional[Config] = None,
    train_config: Optional[RelationTrainConfig] = None,
    category_head: Optional[ClipProjectionCategoryHead] = None,
) -> tuple[LearnedRelationHead, List[float]]:
    import torch

    config = config or Config()
    tcfg = train_config or RelationTrainConfig()
    torch.manual_seed(tcfg.seed)

    if category_head is None:
        category_head = ClipProjectionCategoryHead(
            clip_encoder, vocabulary, temperature=config.head.category_temperature)
        category_head.build_prototypes()

    z_dim = region_encoder.dim
    mlp = build_relation_mlp(z_dim, hidden=tcfg.hidden)
    optim = torch.optim.Adam(mlp.parameters(), lr=tcfg.lr)

    samples = list(dataset)
    history: List[float] = []
    for epoch in range(tcfg.epochs):
        epoch_loss, n_batches = 0.0, 0
        for s_idx, sample in enumerate(samples):
            cands = candidates_from_gt(sample, config, seed=tcfg.seed * 100 + s_idx)
            if len(cands) <= 1:
                continue
            box_crops = [c.crops["box"] for c in cands]
            region_feats = region_encoder.encode(box_crops)
            category_probs = category_head.predict(box_crops)
            ctx = build_pairwise_context(cands, region_feats, category_probs, sample.image.shape)

            match = match_candidates_to_gt([c.mask for c in cands], sample.instances, config.match)
            pairs = sample_pairs(match, ctx, max_pairs=tcfg.max_pairs, seed=tcfg.seed + s_idx)
            if not pairs:
                continue
            targets = relation_pair_targets(match, ctx.containment, pairs)
            phi = build_phi_pairs(ctx, pairs)

            x = torch.from_numpy(phi)
            logits = mlp(x)  # [P, 3] order: sem, inst, part
            y = torch.from_numpy(np.stack(
                [targets["y_sem"], targets["y_inst"], targets["y_part"]], axis=1))
            w = torch.from_numpy(targets["weight"]).clamp_min(1e-3)  # keep gradient alive
            wexp = w[:, None].expand_as(logits)
            loss = weighted_bce_with_logits(logits, y, wexp)

            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))

    head = LearnedRelationHead(mlp, z_dim=z_dim, device="cpu")
    return head, history
