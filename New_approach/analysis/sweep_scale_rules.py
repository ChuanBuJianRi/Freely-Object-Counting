#!/usr/bin/env python3
"""Offline sweep of parameter-free scale-layer count rules on FSC147 test.

Reads pre-dumped per-image blob sigma histograms (``dump_sigma_hist.py``) and
optionally the full-pipeline per-image preds (top1/scale/oracle), joins by image
name, and evaluates several parameter-free scale-selection rules against GT.

Key finding (full test, n=1190):
  most_stable (old)     MAE 47.37  RMSE 131.91
  coarsest_plateau (new) MAE 44.02 RMSE 129.95   <- shipped in scale_count.py
  oracle_scale (cheat)   MAE 15.64 RMSE  66.69

Usage:
  # 1. dump sigma histograms (8-way CPU shards) into outputs/sigma_hist/
  for o in 0 1 2 3 4 5 6 7; do \
    python analysis/dump_sigma_hist.py --stride 8 --offset $o \
      --out outputs/sigma_hist/shard_$o.json & done; wait
  # 2. sweep
  python analysis/sweep_scale_rules.py
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

PROJECT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach")


def load_sigmas(pattern):
    rows = []
    for f in sorted(glob.glob(str(PROJECT / pattern))):
        rows.extend(json.load(open(f)))
    return rows


def curve(sigmas):
    s = np.round(np.array(sigmas, float), 4)
    lv = np.array(sorted(set(s.tolist())))
    c = np.array([int((s == l).sum()) for l in lv], float)
    return lv, c


def most_stable(lv, c):
    n = len(c)
    if n == 0:
        return 0
    if n <= 2:
        return c[-1]
    lvar = [abs(c[i - 1] - c[i]) + abs(c[i] - c[i + 1]) for i in range(1, n - 1)]
    return c[int(np.argmin(lvar)) + 1]


def coarsest_plateau(lv, c):
    n = len(c)
    if n <= 2:
        return c[-1] if n else 0
    rel = np.abs(c[1:] - c[:-1]) / (0.5 * (c[1:] + c[:-1]) + 1.0)
    med = float(np.median(rel))
    plateau = np.where(rel <= med)[0]
    if plateau.size == 0:
        return c[-1]
    return c[min(plateau[-1] + 1, n - 1)]


def oracle_scale(lv, c, gt):
    return min(c, key=lambda v: abs(v - gt)) if len(c) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma_glob", default="outputs/sigma_hist/shard_*.json")
    args = ap.parse_args()

    rows = load_sigmas(args.sigma_glob)
    gts = np.array([r["gt"] for r in rows], float)
    curves = [curve(r["sigmas"]) for r in rows]
    buckets = [(1, 10), (11, 50), (51, 200), (201, 1e9)]

    def report(name, preds):
        preds = np.array(preds, float)
        bm = []
        for lo, hi in buckets:
            m = (gts >= lo) & (gts <= hi)
            bm.append(np.abs(preds[m] - gts[m]).mean() if m.any() else 0.0)
        mae = float(np.abs(preds - gts).mean())
        rmse = float(np.sqrt(((preds - gts) ** 2).mean()))
        print(f"{name:>22} {mae:8.2f} {rmse:9.2f}  "
              f"[{bm[0]:.0f}/{bm[1]:.0f}/{bm[2]:.0f}/{bm[3]:.0f}]")

    print(f"=== scale-layer rule sweep (n={len(rows)}) ===")
    print(f"{'rule':>22} {'MAE':>8} {'RMSE':>9}  buckets[1-10/11-50/51-200/201+]")
    report("most_stable(old)", [most_stable(lv, c) for lv, c in curves])
    report("coarsest_plateau", [coarsest_plateau(lv, c) for lv, c in curves])
    report("oracle_scale(cheat)",
           [oracle_scale(lv, c, g) for (lv, c), g in zip(curves, gts)])


if __name__ == "__main__":
    main()
