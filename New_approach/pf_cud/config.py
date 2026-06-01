"""Engineering configuration only.

Per the PF-CUD design, this module must NOT contain any algorithm thresholds or
tunable inference hyperparameters (no epsilon, delta, k, IoU threshold, FINCH
threshold, number of scales, etc.). It only holds model names, device selection
and paths, which are engineering concerns rather than algorithm tuning knobs.
"""

from dataclasses import dataclass
from typing import Optional

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    """Engineering configuration.

    Attributes:
        device: compute device, e.g. "cuda" or "cpu". None -> auto.
        visual_model_name: backbone name for the visual feature extractor.
        sam_checkpoint: optional path to a SAM/SAM2 checkpoint.
        sam_model_type: optional SAM model type identifier.
    """

    device: Optional[str] = None
    visual_model_name: str = "dinov2_vits14"
    sam_checkpoint: Optional[str] = None
    sam_model_type: Optional[str] = None

    def resolve_device(self) -> str:
        return self.device or default_device()
