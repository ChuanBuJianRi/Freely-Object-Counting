"""A lightweight OCCAM reimplementation based on the paper description.

The official OCCAM code is not public in this repository yet. This package
implements the pipeline described in the paper: SAM2 masks, mask filtering,
ResNet-50 features, and thresholded FINCH-style clustering.
"""

from .config import OccamConfig
from .pipeline import OccamCounter, OccamResult
from .predict import PredictTrace, predict_count

__all__ = [
    "OccamConfig",
    "OccamCounter",
    "OccamResult",
    "predict_count",
    "PredictTrace",
]
