"""Stage 1 training: category projection head (design.md sec 10.3).

Frozen CLIP image encoder + frozen CLIP text prototypes. Trainable parameters:
a projection W (init = identity) and a log-temperature. Loss = weighted CE over
candidate categories (weight = purity * valid).

    h_i       = normalize(e_i @ W)
    logit_i,c = (h_i . t_c) / temperature
    L         = sum_i w_i CE(softmax(logit_i), matched_class_i)   (ignore -1)

Returns the trained ``ClipProjectionCategoryHead`` and the loss history. Works on
the SyntheticGTDataset with deterministic offline CLIP, so it is fully testable
without a real dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import Config
from ..heads.category_head import ClipProjectionCategoryHead
from ..vocabulary import VocabularyBank
from .candidate_source import candidates_from_gt
from .dataset import GTInstanceDataset
from .labels import category_targets
from .losses import weighted_cross_entropy
from .matching import match_candidates_to_gt


@dataclass
class CategoryTrainConfig:
    epochs: int = 3
    lr: float = 1e-3
    seed: int = 0


def train_category(
    dataset: GTInstanceDataset,
    clip_encoder,
    vocabulary: VocabularyBank,
    config: Optional[Config] = None,
    train_config: Optional[CategoryTrainConfig] = None,
) -> tuple[ClipProjectionCategoryHead, List[float]]:
    import torch

    config = config or Config()
    tcfg = train_config or CategoryTrainConfig()
    torch.manual_seed(tcfg.seed)

    # frozen text prototypes [C, D]
    head = ClipProjectionCategoryHead(clip_encoder, vocabulary,
                                      temperature=config.head.category_temperature)
    protos_np = head.build_prototypes()
    protos = torch.from_numpy(protos_np)
    dim = protos.shape[1]

    proj = torch.nn.Linear(dim, dim, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.eye(dim))
    log_temp = torch.nn.Parameter(torch.tensor(float(np.log(config.head.category_temperature))))
    optim = torch.optim.Adam(list(proj.parameters()) + [log_temp], lr=tcfg.lr)

    samples = list(dataset)
    history: List[float] = []
    for epoch in range(tcfg.epochs):
        epoch_loss, n_batches = 0.0, 0
        for s_idx, sample in enumerate(samples):
            cands = candidates_from_gt(sample, config, seed=tcfg.seed * 100 + s_idx)
            if not cands:
                continue
            match = match_candidates_to_gt([c.mask for c in cands], sample.instances, config.match)
            cls_idx, weight = category_targets(match, vocabulary)
            if weight.sum() == 0:
                continue
            embs = clip_encoder.encode_image([c.crops["box"] for c in cands])  # [N, D]
            e = torch.from_numpy(embs)

            h = proj(e)
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = (h @ protos.T) / log_temp.exp().clamp_min(1e-6)
            loss = weighted_cross_entropy(
                logits, torch.from_numpy(cls_idx), torch.from_numpy(weight)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        history.append(epoch_loss / max(1, n_batches))

    # export learned params into the numpy inference head.
    with torch.no_grad():
        head.projection = proj.weight.detach().T.numpy().astype(np.float32)
        head.temperature = float(log_temp.exp().item())
    return head, history
