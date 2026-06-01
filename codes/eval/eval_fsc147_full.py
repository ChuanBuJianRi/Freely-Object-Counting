#!/usr/bin/env python3
"""
Full FSC-147 evaluation for OCCAM (origin_simulation baseline).

Metrics reported per split (val / test):
  MAE   - Mean Absolute Error
  MSE   - Mean Squared Error
  RMSE  - Root Mean Squared Error
  NAE   - Normalized Absolute Error  (AE / GT, images with GT>0)
  rMAE  - relative MAE bucketed by GT range (1-10, 11-50, 51-200, 200+)

Results written to:
  per_image_<split>.json   - per-image detail
  summary.txt              - human-readable summary of all metrics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import subprocess
import torch

torch.set_num_threads(8)
torch.cuda.set_per_process_memory_fraction(0.85, 0)

import numpy as np

from occam import OccamConfig, OccamCounter, predict_count
from occam.pipeline import read_rgb
from occam.sam2_loader import build_sam2_amg, build_sam2_predictor

from _gpu_safety import GpuGuard, add_cli_args as add_gpu_cli_args, guard_from_args

_DEFAULT_FSC147_DIR = Path(__file__).resolve().parents[2] / "datasets" / "FSC147"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = str(PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")


@dataclass
class ImageResult:
    name: str
    gt: int
    pred: int
    ae: int
    se: float
    nae: float
    elapsed: float
    trace: dict | None = None


def compute_metrics(results: list[ImageResult]) -> dict:
    if not results:
        return {}

    gts = np.array([r.gt for r in results], dtype=float)
    preds = np.array([r.pred for r in results], dtype=float)
    aes = np.abs(preds - gts)
    ses = (preds - gts) ** 2

    mae = float(aes.mean())
    mse = float(ses.mean())
    rmse = float(np.sqrt(mse))

    valid = gts > 0
    nae = float((aes[valid] / gts[valid]).mean()) if valid.any() else float("nan")

    buckets = {
        "1-10": (gts >= 1) & (gts <= 10),
        "11-50": (gts >= 11) & (gts <= 50),
        "51-200": (gts >= 51) & (gts <= 200),
        "201+": gts > 200,
    }
    rmae_buckets = {}
    for label, mask in buckets.items():
        if mask.any():
            rmae_buckets[label] = {
                "n": int(mask.sum()),
                "mae": float(aes[mask].mean()),
                "rmse": float(np.sqrt(ses[mask].mean())),
            }

    avg_time = float(np.mean([r.elapsed for r in results]))

    return {
        "n": len(results),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "nae": round(nae, 4) if not np.isnan(nae) else None,
        "avg_time_sec": round(avg_time, 2),
        "by_gt_range": rmae_buckets,
    }


def run_split(
    split_name: str,
    image_names: list[str],
    annotations: dict,
    counter: OccamCounter,
    img_dir: Path,
    output_dir: Path,
    pred_strategy: str = "total",
    mcv_k: float = 1.5,
    mcv_min_anchor_size: int = 0,
    guard: GpuGuard | None = None,
) -> list[ImageResult]:
    results: list[ImageResult] = []
    n = len(image_names)

    print(f"\n{'='*60}")
    print(f"  Split: {split_name}  ({n} images)")
    print(f"{'='*60}")

    for i, name in enumerate(image_names, 1):
        img_path = img_dir / name
        if not img_path.exists():
            print(f"  [{i:4d}/{n}] SKIP (file missing): {name}")
            continue

        ann = annotations.get(name, {})
        gt = len(ann.get("points", []))

        t0 = time.time()
        try:
            image = read_rgb(img_path)
            result = counter.count(image)
        except Exception as exc:
            print(f"  [{i:4d}/{n}] ERROR {name}: {exc}")
            torch.cuda.empty_cache()
            continue
        elapsed = time.time() - t0

        pred, trace = predict_count(
            result,
            pred_strategy,
            image_shape=(int(image.shape[0]), int(image.shape[1])),
            k=mcv_k,
            mcv_min_anchor_size=mcv_min_anchor_size,
        )
        ae = abs(pred - gt)
        se = float((pred - gt) ** 2)
        nae = float(ae / gt) if gt > 0 else float("nan")

        r = ImageResult(
            name=name,
            gt=gt,
            pred=pred,
            ae=ae,
            se=se,
            nae=nae,
            elapsed=round(elapsed, 2),
            trace=trace.to_dict(),
        )
        results.append(r)

        torch.cuda.empty_cache()
        if guard is not None:
            guard.maybe_throttle(i, n)

        if i % 50 == 0 or i == n:
            elapsed_total = sum(x.elapsed for x in results)
            mae_so_far = sum(x.ae for x in results) / len(results)
            print(
                f"  [{i:4d}/{n}]  running MAE={mae_so_far:.2f}  "
                f"total_time={elapsed_total:.0f}s  last: {name} GT={gt} PRED={pred}"
            )
        elif i <= 5 or ae > 80:
            print(f"  [{i:4d}/{n}]  {name:30s} GT={gt:4d} PRED={pred:4d} AE={ae:4d}  {elapsed:.1f}s")

    out_path = output_dir / f"per_image_{split_name}.json"
    out_path.write_text(
        json.dumps(
            [
                {
                    "name": r.name,
                    "gt": r.gt,
                    "pred": r.pred,
                    "ae": r.ae,
                    "se": r.se,
                    "nae": r.nae if not (isinstance(r.nae, float) and np.isnan(r.nae)) else None,
                    "elapsed_sec": r.elapsed,
                    "trace": r.trace,
                }
                for r in results
            ],
            indent=2,
        )
    )
    print(f"\n  Per-image detail saved → {out_path}")
    return results


def write_summary(
    all_metrics: dict[str, dict],
    output_dir: Path,
    mode: str,
    started_at: str,
    finished_at: str,
    total_elapsed: float,
    args_fraction: float = 1.0,
    args_seed: int = 42,
) -> None:
    lines = []
    lines.append("=" * 70)
    lines.append("  OCCAM Origin-Simulation — FSC-147 Full Evaluation Summary")
    lines.append("=" * 70)
    lines.append(f"  Mode        : OCCAM-{mode.upper()} (fraction={args_fraction:.3f}, seed={args_seed})")
    lines.append(f"  SAM2 config : {SAM2_CONFIG}")
    lines.append(f"  SAM2 ckpt   : {SAM2_CKPT}")
    lines.append(f"  Started     : {started_at}")
    lines.append(f"  Finished    : {finished_at}")
    lines.append(f"  Total time  : {total_elapsed/60:.1f} min")
    lines.append("")

    for split, m in all_metrics.items():
        lines.append(f"  ── {split} split ({'n=' + str(m['n'])})")
        lines.append(f"     MAE   = {m['mae']:.4f}")
        lines.append(f"     MSE   = {m['mse']:.4f}")
        lines.append(f"     RMSE  = {m['rmse']:.4f}")
        nae_str = f"{m['nae']:.4f}" if m.get("nae") is not None else "N/A"
        lines.append(f"     NAE   = {nae_str}")
        lines.append(f"     avg_time = {m['avg_time_sec']:.2f} s/img")
        if m.get("by_gt_range"):
            lines.append("     by GT count range:")
            for rng, bm in m["by_gt_range"].items():
                lines.append(
                    f"       [{rng:>7s}]  n={bm['n']:4d}  MAE={bm['mae']:7.2f}  RMSE={bm['rmse']:7.2f}"
                )
        lines.append("")

    lines.append("=" * 70)

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSummary saved → {summary_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("single", "multi"), default="single")
    p.add_argument("--device", default="cuda")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   help="Which splits to evaluate (val, test)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit images per split (for quick debug)")
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Random fraction of each split to evaluate (e.g. 0.333 for 1/3)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for fraction sampling")
    p.add_argument("--points-per-side", type=int, default=32)
    p.add_argument("--pred-iou-thresh", type=float, default=0.7)
    p.add_argument("--stability-thresh", type=float, default=0.92)
    p.add_argument("--data-dir", type=str, default=None,
                   help="FSC-147 dataset root (default: <repo_root>/datasets/FSC147)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for results (default: same as script)")
    p.add_argument("--min-mask-area", type=float, default=None,
                   help="Override min_mask_area_ratio in OccamConfig")
    p.add_argument("--max-mask-area", type=float, default=None,
                   help="Override max_mask_area_ratio in OccamConfig")
    p.add_argument("--cluster-method", choices=("finch", "sng"), default="finch",
                   help="Clustering algorithm")
    p.add_argument("--sng-epsilon", type=int, default=10,
                   help="SNG: each node connects to its epsilon nearest neighbours")
    p.add_argument("--sng-delta", type=int, default=None,
                   help="SNG: edges with common neighbours <= delta are removed. "
                        "Default None ⇒ auto-derive via §7.1 adaptive rule (see --sng-alpha).")
    p.add_argument("--sng-alpha", type=float, default=0.4,
                   help="SNG adaptive-delta blend coefficient: "
                        "delta* = floor(alpha*(eps-1) + (1-alpha)*eps^2/n). "
                        "Sweet spot alpha in [0.3, 0.5]; ignored when --sng-delta is set.")
    p.add_argument("--pred-strategy",
                   choices=("total", "max", "mode_cluster_vote", "mcv"),
                   default="total",
                   help="PRED = sum of all cluster sizes (total) or largest cluster size (max) "
                        "or mode-cluster-vote / mcv (sum every cluster within k*MAD log-area "
                        "of the largest cluster's log-area). See library/notes/MCV-method.md.")
    p.add_argument("--mask-policy",
                   choices=("p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"),
                   default="p0",
                   help="Mask filtering policy: p0 area_window | p1 score_thresh | "
                        "p2 topk | p3 area_iqr | p4 area_otsu | p5 score_and_area | "
                        "p6 score_nms | p7 no_filter")
    p.add_argument("--mask-score-thresh", type=float, default=0.85,
                   help="SAM2 predicted_iou threshold for p1/p5")
    p.add_argument("--mask-topk", type=int, default=100,
                   help="Top-K masks by SAM2 score for p2")
    p.add_argument("--mask-iqr-k", type=float, default=1.5,
                   help="IQR multiplier for p3 (Q1-k*IQR, Q3+k*IQR)")
    p.add_argument("--mcv-min-anchor-size", type=int, default=0,
                   help="MCV-only guard: when the anchor (largest non-singleton) "
                        "cluster has fewer than this many members, fall back to 'max'. "
                        "Default 0 disables the guard (original MCV behaviour).")
    p.add_argument("--mask-backend", choices=("amg", "predictor"), default="amg",
                   help="Mask generation backend. 'amg' = SAM2AutomaticMaskGenerator "
                        "(32x32 grid, internal NMS). 'predictor' = OCCAM-paper dense "
                        "seed-grid prompting (spacing px) + paper mask processing "
                        "(largest-CC, drop 1px / oversized, IoU dedup).")
    p.add_argument("--seed-spacing", type=int, default=10,
                   help="Predictor backend: seed-point grid spacing in pixels "
                        "(OCCAM paper uses 10). Ignored by AMG backend.")
    p.add_argument("--duplicate-iou", type=float, default=None,
                   help="Duplicate-mask IoU threshold. OCCAM paper uses 0.1; our "
                        "AMG default is 0.5. Overrides OccamConfig.duplicate_iou_threshold.")
    p.add_argument("--enable-multiscale", action="store_true",
                   help="Enable the OCCAM 3x3 multiscale refinement (paper 'Scaling' "
                        "component; Table 8 shows it improves MAE/RMSE). Default off.")
    add_gpu_cli_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    fsc147_dir = Path(args.data_dir) if args.data_dir else _DEFAULT_FSC147_DIR
    output_dir = Path(args.output_dir) if args.output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = json.loads((fsc147_dir / "annotation_FSC147_384.json").read_text())
    split_info = json.loads((fsc147_dir / "Train_Test_Val_FSC_147.json").read_text())
    img_dir = fsc147_dir / "images_384_VarV2"

    if args.mask_backend == "predictor":
        print("Building SAM2 Predictor (paper seed-grid path) …")
        predictor = build_sam2_predictor(
            model_config=SAM2_CONFIG,
            checkpoint=SAM2_CKPT,
            device=args.device,
        )
        amg = None
    else:
        print("Building SAM2 AMG model …")
        amg = build_sam2_amg(
            model_config=SAM2_CONFIG,
            checkpoint=SAM2_CKPT,
            device=args.device,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_thresh,
        )
        predictor = None
    config = OccamConfig.for_mode(
        args.mode,
        device=args.device,
        enable_multiscale=args.enable_multiscale,
        min_mask_area_ratio=args.min_mask_area,
        max_mask_area_ratio=args.max_mask_area,
        cluster_method=args.cluster_method,
        sng_epsilon=args.sng_epsilon,
        sng_delta=args.sng_delta,
        sng_alpha=args.sng_alpha,
        pred_strategy=args.pred_strategy,
        mask_policy=args.mask_policy,
        mask_score_thresh=args.mask_score_thresh,
        mask_topk=args.mask_topk,
        mask_iqr_k=args.mask_iqr_k,
        seed_spacing=args.seed_spacing,
        duplicate_iou_threshold=args.duplicate_iou,
    )
    if predictor is not None:
        counter = OccamCounter(config, predictor=predictor)
    else:
        counter = OccamCounter(config, amg=amg)
    print(f"Model ready. Backend={args.mask_backend}, Mode={args.mode}, "
          f"crop={config.crop_size}, finch_thresholds={config.finch_thresholds}, "
          f"multiscale={config.enable_multiscale}, dup_iou={config.duplicate_iou_threshold}, "
          f"seed_spacing={config.seed_spacing}")
    print(f"Mask area filter: min={config.min_mask_area_ratio}, max={config.max_mask_area_ratio}")
    print(f"Mask policy: {config.mask_policy}  "
          f"(score_thresh={config.mask_score_thresh}, topk={config.mask_topk}, "
          f"iqr_k={config.mask_iqr_k})")
    print(f"Cluster method: {config.cluster_method} "
          f"(eps={config.sng_epsilon}, delta={config.sng_delta} for sng)  "
          f"PRED strategy: {config.pred_strategy}")

    guard = guard_from_args(args)
    if not guard.enabled and args.device.startswith("cuda") and not args.gpu_guard_off:
        print("[gpu-guard] WARN: guard disabled but device is CUDA. "
              "Continuing without thermal protection.")
    if args.gpu_guard_off and args.device.startswith("cuda"):
        print("[gpu-guard] WARN: --gpu-guard-off explicitly set on a CUDA run. "
              "This violates project policy unless this is a CPU-only smoke test.")

    global_start = time.time()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    all_metrics: dict[str, dict] = {}

    rng = np.random.default_rng(args.seed)

    for split in args.splits:
        image_names = split_info.get(split, [])
        if args.fraction < 1.0:
            n_sample = max(1, int(round(len(image_names) * args.fraction)))
            image_names = list(rng.choice(image_names, size=n_sample, replace=False))
            image_names.sort()
        if args.limit:
            image_names = image_names[: args.limit]

        results = run_split(split, image_names, annotations, counter, img_dir, output_dir,
                            pred_strategy=args.pred_strategy, mcv_k=args.mask_iqr_k,
                            mcv_min_anchor_size=args.mcv_min_anchor_size, guard=guard)
        metrics = compute_metrics(results)
        all_metrics[split] = metrics

        print(f"\n  [{split}] MAE={metrics['mae']}  MSE={metrics['mse']}  "
              f"RMSE={metrics['rmse']}  NAE={metrics.get('nae')}")

    finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
    total_elapsed = time.time() - global_start

    metrics_path = output_dir / "metrics.json"
    payload = dict(all_metrics)
    payload["thermal"] = guard.to_dict()
    metrics_path.write_text(json.dumps(payload, indent=2))

    write_summary(all_metrics, output_dir, args.mode, started_at, finished_at, total_elapsed,
                  args_fraction=args.fraction, args_seed=args.seed)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
