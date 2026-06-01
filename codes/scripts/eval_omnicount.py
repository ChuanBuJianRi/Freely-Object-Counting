#!/usr/bin/env python3
"""Quick OCCAM evaluation on a slice of OmniCount-191.

Loads the COCO-style annotations next to a folder of test images and runs the
counter on the first ``--limit`` images, reporting per-image GT total versus
predicted total plus a final MAE summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter
from dataclasses import replace

from occam import OccamConfig, OccamCounter
from occam.pipeline import draw_result, read_rgb, write_rgb
from occam.sam2_loader import build_sam2_amg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", required=True, help="OmniCount _annotations.coco.json")
    parser.add_argument("--image-dir", required=True, help="Directory of test images")
    parser.add_argument("--sam2-config", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("single", "multi"), default="single")
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-thresh", type=float, default=0.92)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coco = json.loads(Path(args.coco_json).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    image_index = {img["id"]: img for img in coco["images"]}
    gt_counts: dict[int, Counter] = {}
    for ann in coco["annotations"]:
        gt_counts.setdefault(ann["image_id"], Counter())[cats[ann["category_id"]]] += 1

    selected = list(coco["images"])[: args.limit]

    amg = build_sam2_amg(
        model_config=args.sam2_config,
        checkpoint=args.sam2_checkpoint,
        device=args.device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
    )
    config = OccamConfig.for_mode(args.mode, device=args.device)
    counter = OccamCounter(config, amg=amg)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    abs_errors_total: list[int] = []
    abs_errors_classes: list[int] = []
    rows: list[dict] = []

    for image_meta in selected:
        image_id = image_meta["id"]
        gt_per_class = gt_counts.get(image_id, Counter())
        gt_total = sum(gt_per_class.values())
        gt_classes = len(gt_per_class)

        image_path = Path(args.image_dir) / image_meta["file_name"]
        if not image_path.exists():
            print(f"[skip] missing {image_path}")
            continue

        image = read_rgb(image_path)
        start = time.time()
        result = counter.count(image)
        elapsed = time.time() - start

        pred_total = result.total_count
        pred_classes = len(result.clusters)

        abs_errors_total.append(abs(pred_total - gt_total))
        abs_errors_classes.append(abs(pred_classes - gt_classes))

        rows.append(
            {
                "image": image_meta["file_name"],
                "gt_total": gt_total,
                "pred_total": pred_total,
                "gt_classes": gt_classes,
                "pred_classes": pred_classes,
                "gt_per_class": dict(gt_per_class),
                "pred_counts": result.counts,
                "elapsed_sec": round(elapsed, 2),
            }
        )

        print(
            f"{image_meta['file_name'][:60]:60s} "
            f"GT={gt_total} ({gt_classes}cls)  PRED={pred_total} ({pred_classes}cls)  "
            f"{elapsed:5.1f}s"
        )

        if output_dir is not None:
            visualization = draw_result(image, result)
            write_rgb(output_dir / f"{Path(image_meta['file_name']).stem}.jpg", visualization)

    if abs_errors_total:
        mae_total = sum(abs_errors_total) / len(abs_errors_total)
        mae_classes = sum(abs_errors_classes) / len(abs_errors_classes)
        summary = {
            "n_images": len(rows),
            "mae_total_count": round(mae_total, 2),
            "mae_class_count": round(mae_classes, 2),
            "rows": rows,
        }
        print("\n=== summary ===")
        print(json.dumps({k: summary[k] for k in ("n_images", "mae_total_count", "mae_class_count")}, indent=2))
        if output_dir is not None:
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
