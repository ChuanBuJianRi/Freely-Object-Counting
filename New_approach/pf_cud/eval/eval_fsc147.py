"""Evaluate PF-CUD on the FSC147 test split.

FSC147 is a class-agnostic counting benchmark. PF-CUD is prior-free (no exemplar
boxes, no text prompt), so it outputs a set of ranked counting hypotheses. We
report two predictors:

- top1   : count of the highest-ranked group (the method's actual prediction).
- oracle : count of the group closest to GT (upper bound given the candidate
           grouping, useful to separate ranking error from grouping error).

CLI exposes only engineering options (dataset paths, subset size, device) plus
the project-standard GPU thermal-safety flags (--gpu-*). No algorithm tuning
parameters are exposed, consistent with the parameter-free design.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from pf_cud.eval.metrics import mae, nae, rmse, sre
from pf_cud.pipeline import PFCUDPipeline
from pf_cud.select.group_filter import select_counting_groups
from pf_cud.select.scale_count import scale_layer_count_from_sigmas

# Reuse the project-wide GPU thermal guard (bench/codes/eval/_gpu_safety.py) so
# this evaluator complies with results/index.md GPU_THERMAL_POLICY.
_CODES_EVAL_DIR = Path(__file__).resolve().parents[3] / "codes" / "eval"
if str(_CODES_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_CODES_EVAL_DIR))
from _gpu_safety import add_cli_args as add_gpu_cli_args  # noqa: E402
from _gpu_safety import guard_from_args  # noqa: E402

# Default dataset root on this machine. Override with --dataset_root.
_DEFAULT_DATASET_ROOT = "/home/gaoyiyang/ws_yiyang/datasets/FSC147"


def load_split(dataset_root: str, split: str) -> List[str]:
    with open(os.path.join(dataset_root, "Train_Test_Val_FSC_147.json")) as f:
        splits = json.load(f)
    return splits[split]


def load_annotations(dataset_root: str) -> dict:
    with open(os.path.join(dataset_root, "annotation_FSC147_384.json")) as f:
        return json.load(f)


def bucket_metrics(preds: List[int], gts: List[int]) -> dict:
    preds_a = np.asarray(preds, dtype=float)
    gts_a = np.asarray(gts, dtype=float)
    aes = np.abs(preds_a - gts_a)
    ses = (preds_a - gts_a) ** 2
    buckets = {
        "1-10": (gts_a >= 1) & (gts_a <= 10),
        "11-50": (gts_a >= 11) & (gts_a <= 50),
        "51-200": (gts_a >= 51) & (gts_a <= 200),
        "201+": gts_a > 200,
    }
    out = {}
    for label, mask in buckets.items():
        if mask.any():
            out[label] = {
                "n": int(mask.sum()),
                "mae": float(aes[mask].mean()),
                "rmse": float(np.sqrt(ses[mask].mean())),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description="PF-CUD FSC147 evaluation.")
    parser.add_argument("--dataset_root", default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    parser.add_argument("--stride", type=int, default=1, help="subsample split")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="start index before striding; used to shard a split across "
        "multiple GPUs (e.g. stride=4 offset=0..3). Engineering only.",
    )
    parser.add_argument("--out_json", default="outputs/fsc147_eval.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use_edge", action="store_true")
    parser.add_argument("--no_visual", action="store_true", help="skip DINOv2 (Phase 1)")
    add_gpu_cli_args(parser)
    args = parser.parse_args()

    images = load_split(args.dataset_root, args.split)
    ann = load_annotations(args.dataset_root)
    img_dir = os.path.join(args.dataset_root, "images_384_VarV2")

    images = images[args.offset :: args.stride]
    if args.limit > 0:
        images = images[: args.limit]

    pipeline = PFCUDPipeline(
        sam_model=None, use_edge=args.use_edge, use_visual=not args.no_visual
    )

    guard = guard_from_args(args)

    gt_counts: List[int] = []
    top1_counts: List[int] = []
    oracle_counts: List[int] = []
    select_counts: List[int] = []
    scale_counts: List[int] = []
    per_image = []

    n = len(images)
    t0 = time.time()
    for i, name in enumerate(images):
        gt = len(ann[name]["points"])
        path = os.path.join(img_dir, name)
        image = np.array(Image.open(path).convert("RGB"))

        img_t0 = time.time()
        result = pipeline.run(image)
        elapsed_img = time.time() - img_t0
        group_counts = [len(g.indices) for g in result.groups]

        if group_counts:
            top1 = group_counts[0]
            oracle = min(group_counts, key=lambda c: abs(c - gt))
        else:
            top1 = 0
            oracle = 0

        # Adaptive parameter-free counting-group selector: keep the countable
        # foreground groups, rank them by (repetition, total coverage), take the
        # dominant one. Reported alongside top1/oracle to gauge its effect.
        selected = select_counting_groups(result.candidates, result.groups,
                                           result.image_shape)
        select_pred = len(selected[0].indices) if selected else 0

        # Scale-layer counting: read the count off the most stable blob scale
        # level (uses the pre-dedup sigma histogram captured in result.meta).
        scale_pred = scale_layer_count_from_sigmas(
            result.meta.get("raw_blob_sigmas", [])
        )

        gt_counts.append(gt)
        top1_counts.append(top1)
        oracle_counts.append(oracle)
        select_counts.append(select_pred)
        scale_counts.append(scale_pred)
        per_image.append(
            {
                "image": name,
                "gt": gt,
                "top1": top1,
                "oracle": oracle,
                "select": select_pred,
                "scale": scale_pred,
                "num_candidates": len(result.candidates),
                "num_groups": len(result.groups),
                "elapsed_sec": round(elapsed_img, 2),
            }
        )

        # Thermal guard (polls nvidia-smi every --gpu-check-every images).
        guard.maybe_throttle(i + 1, n)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            mae_so_far = mae(top1_counts, gt_counts)
            sel_mae = mae(select_counts, gt_counts)
            scl_mae = mae(scale_counts, gt_counts)
            print(
                f"[{i + 1}/{n}] {name} gt={gt} top1={top1} "
                f"oracle={oracle} select={select_pred} scale={scale_pred} "
                f"cand={len(result.candidates)} "
                f"groups={len(result.groups)} top1_MAE={mae_so_far:.2f} "
                f"select_MAE={sel_mae:.2f} scale_MAE={scl_mae:.2f} "
                f"({elapsed:.1f}s, {elapsed_img:.1f}s/img)",
                flush=True,
            )

    metrics = {
        "num_images": n,
        "top1_mae": mae(top1_counts, gt_counts),
        "top1_rmse": rmse(top1_counts, gt_counts),
        "top1_nae": nae(top1_counts, gt_counts),
        "top1_sre": sre(top1_counts, gt_counts),
        "select_mae": mae(select_counts, gt_counts),
        "select_rmse": rmse(select_counts, gt_counts),
        "select_nae": nae(select_counts, gt_counts),
        "select_sre": sre(select_counts, gt_counts),
        "scale_mae": mae(scale_counts, gt_counts),
        "scale_rmse": rmse(scale_counts, gt_counts),
        "scale_nae": nae(scale_counts, gt_counts),
        "scale_sre": sre(scale_counts, gt_counts),
        "oracle_mae": mae(oracle_counts, gt_counts),
        "oracle_rmse": rmse(oracle_counts, gt_counts),
        "top1_by_gt_range": bucket_metrics(top1_counts, gt_counts),
        "select_by_gt_range": bucket_metrics(select_counts, gt_counts),
        "scale_by_gt_range": bucket_metrics(scale_counts, gt_counts),
        "oracle_by_gt_range": bucket_metrics(oracle_counts, gt_counts),
        "elapsed_sec": time.time() - t0,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "thermal": guard.to_dict(), "per_image": per_image},
            f,
            indent=2,
        )

    print("\n=== FSC147", args.split, "results ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
