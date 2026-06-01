"""SAM2 loader utilities.

SAM2 is intentionally kept as an optional dependency because checkpoint and
config paths vary by installation.
"""

from __future__ import annotations


def _import_sam2():
    try:
        from sam2.build_sam import build_sam2
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. Install it from "
            "https://github.com/facebookresearch/sam2 and pass "
            "--sam2-config/--sam2-checkpoint."
        ) from exc
    return build_sam2


def build_sam2_predictor(
    *,
    model_config: str,
    checkpoint: str,
    device: str,
):
    """Build a ``SAM2ImagePredictor`` (single-point prompting mode)."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    build_sam2 = _import_sam2()
    model = build_sam2(model_config, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def build_sam2_amg(
    *,
    model_config: str,
    checkpoint: str,
    device: str,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.92,
    box_nms_thresh: float = 0.7,
    min_mask_region_area: int = 100,
    crop_n_layers: int = 1,
    crop_nms_thresh: float = 0.7,
):
    """Build a ``SAM2AutomaticMaskGenerator`` for dense mask generation.

    The default parameters are tuned to produce higher-recall, lower-noise
    masks compared to the SAM2 defaults, matching the OCCAM paper's intent of
    capturing every individual object in the image.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    build_sam2 = _import_sam2()
    model = build_sam2(model_config, checkpoint, device=device)
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        output_mode="binary_mask",
    )
