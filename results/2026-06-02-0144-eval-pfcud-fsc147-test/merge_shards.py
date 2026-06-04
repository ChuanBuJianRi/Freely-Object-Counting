#!/usr/bin/env python3
"""Merge PF-CUD FSC-147 sharded eval outputs into final metrics.json + summary.

Reads per_image_test_shard{0..3}.json from the run dir, concatenates the
per-image records, recomputes overall top1/oracle metrics (MAE/RMSE/NAE/SRE +
by-GT-range buckets), and aggregates the 4 thermal guard blocks (peak temp =
max, cooldown events/seconds/polls = sum). Writes metrics.json, per_image_test.json
and summary.txt.
"""

import glob
import json
import os
import sys

import numpy as np

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def mae(p, g):
    return float(np.mean(np.abs(np.asarray(p, float) - np.asarray(g, float))))


def rmse(p, g):
    return float(np.sqrt(np.mean((np.asarray(p, float) - np.asarray(g, float)) ** 2)))


def nae(p, g):
    p, g = np.asarray(p, float), np.asarray(g, float)
    return float(np.mean(np.abs(p - g) / np.maximum(1.0, g)))


def sre(p, g):
    p, g = np.asarray(p, float), np.asarray(g, float)
    return float(np.mean((p - g) ** 2 / np.maximum(1.0, g)))


def buckets(p, g):
    p, g = np.asarray(p, float), np.asarray(g, float)
    aes, ses = np.abs(p - g), (p - g) ** 2
    defs = {
        "1-10": (g >= 1) & (g <= 10),
        "11-50": (g >= 11) & (g <= 50),
        "51-200": (g >= 51) & (g <= 200),
        "201+": g > 200,
    }
    out = {}
    for lbl, m in defs.items():
        if m.any():
            out[lbl] = {
                "n": int(m.sum()),
                "mae": round(float(aes[m].mean()), 4),
                "rmse": round(float(np.sqrt(ses[m].mean())), 4),
            }
    return out


def main():
    shard_files = sorted(glob.glob(os.path.join(RUN_DIR, "per_image_test_shard*.json")))
    if not shard_files:
        sys.exit(f"no shard files in {RUN_DIR}")

    per_image = []
    thermals = []
    for f in shard_files:
        d = json.load(open(f))
        per_image.extend(d["per_image"])
        thermals.append(d["thermal"])

    per_image.sort(key=lambda r: r["image"])
    gt = [r["gt"] for r in per_image]
    top1 = [r["top1"] for r in per_image]
    oracle = [r["oracle"] for r in per_image]

    metrics = {
        "num_images": len(per_image),
        "top1_mae": round(mae(top1, gt), 4),
        "top1_rmse": round(rmse(top1, gt), 4),
        "top1_nae": round(nae(top1, gt), 4),
        "top1_sre": round(sre(top1, gt), 4),
        "oracle_mae": round(mae(oracle, gt), 4),
        "oracle_rmse": round(rmse(oracle, gt), 4),
        "top1_by_gt_range": buckets(top1, gt),
        "oracle_by_gt_range": buckets(oracle, gt),
        "avg_time_sec": round(float(np.mean([r["elapsed_sec"] for r in per_image])), 2),
    }

    thermal = {
        "enabled": any(t["enabled"] for t in thermals),
        "temp_limit_c": thermals[0]["temp_limit_c"],
        "cooldown_sec": thermals[0]["cooldown_sec"],
        "hysteresis_c": thermals[0]["hysteresis_c"],
        "check_every": thermals[0]["check_every"],
        "peak_temp_c": max(t["peak_temp_c"] for t in thermals),
        "cooldown_events": sum(t["cooldown_events"] for t in thermals),
        "cooldown_seconds": round(sum(t["cooldown_seconds"] for t in thermals), 1),
        "polls": sum(t["polls"] for t in thermals),
        "shards": len(thermals),
        "per_shard_peak_c": [t["peak_temp_c"] for t in thermals],
    }

    json.dump(
        {"metrics": metrics, "thermal": thermal, "per_image": per_image},
        open(os.path.join(RUN_DIR, "per_image_test.json"), "w"),
        indent=2,
    )
    payload = {"test": metrics, "thermal": thermal}
    json.dump(payload, open(os.path.join(RUN_DIR, "metrics.json"), "w"), indent=2)

    lines = [
        "=" * 60,
        "  PF-CUD — FSC-147 test Evaluation (4-GPU sharded)",
        "=" * 60,
        f"  images      : {metrics['num_images']}",
        f"  top1   MAE  : {metrics['top1_mae']:.4f}   RMSE {metrics['top1_rmse']:.4f}"
        f"   NAE {metrics['top1_nae']:.4f}",
        f"  oracle MAE  : {metrics['oracle_mae']:.4f}   RMSE {metrics['oracle_rmse']:.4f}",
        f"  avg_time    : {metrics['avg_time_sec']:.2f} s/img",
        f"  thermal     : peak {thermal['peak_temp_c']}C  "
        f"events {thermal['cooldown_events']}  polls {thermal['polls']}",
        "",
        "  top1 by GT range:",
    ]
    for rng, bm in metrics["top1_by_gt_range"].items():
        lines.append(f"    [{rng:>7s}] n={bm['n']:4d}  MAE={bm['mae']:8.2f}  RMSE={bm['rmse']:8.2f}")
    lines.append("")
    lines.append("  oracle by GT range:")
    for rng, bm in metrics["oracle_by_gt_range"].items():
        lines.append(f"    [{rng:>7s}] n={bm['n']:4d}  MAE={bm['mae']:8.2f}  RMSE={bm['rmse']:8.2f}")
    lines.append("=" * 60)
    txt = "\n".join(lines)
    open(os.path.join(RUN_DIR, "summary.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
