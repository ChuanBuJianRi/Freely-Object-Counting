#!/usr/bin/env python3
"""Run the OCCAM reimplementation on one image."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from occam import OccamConfig, OccamCounter
from occam.pipeline import draw_result, read_rgb, write_rgb
from occam.sam2_loader import build_sam2_amg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Path to an RGB image.")
    parser.add_argument("--sam2-config", required=True, help="SAM2 model config name/path.")
    parser.add_argument("--sam2-checkpoint", required=True, help="SAM2 checkpoint path.")
    parser.add_argument(
        "--mode",
        choices=("single", "multi"),
        default="single",
        help="Use OCCAM-S or OCCAM-M settings.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="AMG points_per_side (denser = more masks, slower).",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.7,
        help="AMG predicted IoU quality threshold.",
    )
    parser.add_argument(
        "--stability-thresh",
        type=float,
        default=0.92,
        help="AMG stability score threshold.",
    )
    parser.add_argument("--output", help="Optional path for a visualization image.")
    parser.add_argument("--json-output", help="Optional path for JSON counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_rgb(args.image)

    amg = build_sam2_amg(
        model_config=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
    )
    config = OccamConfig.for_mode(args.mode, device=args.device)
    config = replace(
        config,
        amg_points_per_side=args.points_per_side,
        amg_pred_iou_thresh=args.pred_iou_thresh,
        amg_stability_score_thresh=args.stability_thresh,
    )

    result = OccamCounter(config, amg=amg).count(image)

    payload = {
        "image": str(Path(args.image)),
        "mode": args.mode,
        "num_clusters": len(result.clusters),
        "counts": result.counts,
        "total_count": result.total_count,
    }
    print(json.dumps(payload, indent=2))

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output:
        visualization = draw_result(image, result)
        write_rgb(args.output, visualization)


if __name__ == "__main__":
    main()
