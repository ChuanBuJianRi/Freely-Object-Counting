#!/usr/bin/env python3
"""Cache PF-CUD group descriptors for a fixed FSC147 sample, then sweep many
parameter-free count-selection scoring variants offline.

Stage A (slow, GPU, once): run the full pipeline on the sample and dump, per
image, per output group, a compact descriptor (count, areas, internal
consistency, per-feature residuals, background flag, spatial spread). Cached to
a .json so scoring variants can be compared in milliseconds.

Stage B (fast, CPU): evaluate several scoring rules on the cache and print
MAE/RMSE vs GT (oracle = closest group, cheating upper bound).

All scoring variants are parameter-free (Otsu / rank / log only).

Usage:
  # build cache (GPU 2), 25 imgs:
  CUDA_VISIBLE_DEVICES=2 python analysis/sweep_select.py --build --stride 47 --limit 25
  # then sweep (no GPU needed):
  python analysis/sweep_select.py --sweep
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

DATASET = "/home/gaoyiyang/ws_yiyang/datasets/FSC147"
CACHE = str(PROJECT / "outputs" / "select_sweep_cache.json")

_BG = "background_or_pattern"


def _feature_residual(members, key):
    feats = [c.features[key] for c in members if key in c.features]
    if len(feats) <= 1:
        return 0.0
    x = np.stack(feats, axis=0)
    mu = x.mean(axis=0, keepdims=True)
    return float(np.mean((x - mu) ** 2))


def build_cache(stride, limit, offset):
    from PIL import Image
    from pf_cud.pipeline import PFCUDPipeline
    from pf_cud.features.fusion import fused_distance

    splits = json.load(open(os.path.join(DATASET, "Train_Test_Val_FSC_147.json")))
    ann = json.load(open(os.path.join(DATASET, "annotation_FSC147_384.json")))
    img_dir = os.path.join(DATASET, "images_384_VarV2")
    names = splits["test"][offset::stride][:limit]

    pipe = PFCUDPipeline(sam_model=None, use_edge=True, use_visual=True)
    out = []
    t0 = time.time()
    for i, name in enumerate(names):
        gt = len(ann[name]["points"])
        img = np.array(Image.open(os.path.join(img_dir, name)).convert("RGB"))
        res = pipe.run(img)
        h, w = res.image_shape
        image_area = float(h * w)
        groups = []
        for g in res.groups:
            members = [res.candidates[k] for k in g.indices]
            count = len(g.indices)
            total_area = sum(int(m.mask.sum()) for m in members) / image_area
            mean_area = total_area / max(1, count)
            # internal consistency = 1 / mean pairwise fused distance
            cons = 0.0
            if count > 1:
                d = fused_distance(members)
                iu = np.triu_indices(d.shape[0], k=1)
                md = float(d[iu].mean()) if iu[0].size else 0.0
                cons = (1.0 / md) if md > 0 else float("inf")
            # bbox centre spread
            centres = np.array(
                [[(m.bbox[0] + m.bbox[2]) / 2.0 / w,
                  (m.bbox[1] + m.bbox[3]) / 2.0 / h] for m in members]
            )
            spread = (
                float(np.linalg.det(np.cov(centres.T) + np.eye(2) * 1e-6))
                if count > 1 else 0.0
            )
            groups.append({
                "count": count,
                "total_area": total_area,
                "mean_area": mean_area,
                "consistency": cons,
                "spread": spread,
                "res_visual": _feature_residual(members, "visual"),
                "res_shape": _feature_residual(members, "shape"),
                "res_color": _feature_residual(members, "color"),
                "is_bg": g.group_type == _BG,
            })
        out.append({"image": name, "gt": gt, "groups": groups})
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(names)}] {name} gt={gt} groups={len(groups)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(out, open(CACHE, "w"))
    print(f"cached {len(out)} images -> {CACHE}")


# ---- Stage B: scoring variants (all parameter-free) ----

def _rank01(vals, higher=True):
    a = np.asarray(vals, float)
    n = len(a)
    if n == 0:
        return a
    if n == 1:
        return np.ones(1)
    fin = a[np.isfinite(a)]
    hi = fin.max() if fin.size else 0.0
    lo = fin.min() if fin.size else 0.0
    a = np.where(np.isposinf(a), hi, a)
    a = np.where(np.isneginf(a), lo, a)
    order = np.argsort(a)
    if higher:
        order = order[::-1]
    r = np.empty(n)
    r[order] = np.arange(n)
    return 1.0 - r / (n - 1)


def _otsu_keep(strength):
    from skimage.filters import threshold_otsu
    u = np.unique(strength)
    if u.size <= 1:
        return np.ones(len(strength), bool)
    tau = threshold_otsu(strength)
    keep = strength >= tau
    return keep if keep.any() else np.ones(len(strength), bool)


def _foreground(groups):
    fg = [g for g in groups if not g["is_bg"]]
    return fg if fg else list(groups)


def variant(groups, score_fn, use_keep=True):
    """Generic: foreground -> optional Otsu keep on score -> argmax score."""
    fg = _foreground(groups)
    if not fg:
        return 0
    s = score_fn(fg)
    if use_keep and len(fg) > 1:
        keep = _otsu_keep(s)
        fg2 = [g for g, k in zip(fg, keep) if k]
        s2 = score_fn(fg2)
    else:
        fg2, s2 = fg, s
    if not fg2:
        return 0
    return fg2[int(np.argmax(s2))]["count"]


def s_logcount_cons(fg):
    rep = _rank01([np.log1p(g["count"]) for g in fg])
    cons = _rank01([g["consistency"] for g in fg])
    return 0.5 * (rep + cons)


def s_count_cons(fg):  # raw count (not log) + consistency
    rep = _rank01([g["count"] for g in fg])
    cons = _rank01([g["consistency"] for g in fg])
    return 0.5 * (rep + cons)


def s_cons_only(fg):
    return _rank01([g["consistency"] for g in fg])


def s_count_only(fg):
    return _rank01([g["count"] for g in fg])


def s_count2_cons(fg):  # weight repetition 2x consistency
    rep = _rank01([np.log1p(g["count"]) for g in fg])
    cons = _rank01([g["consistency"] for g in fg])
    return (2 * rep + cons) / 3.0


def s_count_cons_lowres(fg):
    # repetition + consistency + low feature residual (visual+shape+color)
    rep = _rank01([np.log1p(g["count"]) for g in fg])
    cons = _rank01([g["consistency"] for g in fg])
    res = _rank01([g["res_visual"] + g["res_shape"] + g["res_color"] for g in fg],
                  higher=False)
    return (rep + cons + res) / 3.0


def s_count_meanarea(fg):
    # repetition + small-ish single-object size (moderate mean area)
    rep = _rank01([np.log1p(g["count"]) for g in fg])
    # prefer mean area close to the median (objects, not specks or whole-image)
    ma = np.array([g["mean_area"] for g in fg])
    med = np.median(ma)
    moderate = _rank01(np.abs(ma - med), higher=False)
    return 0.5 * (rep + moderate)


VARIANTS = {
    "logcount+cons": s_logcount_cons,
    "count+cons": s_count_cons,
    "cons_only": s_cons_only,
    "count_only": s_count_only,
    "2logcount+cons": s_count2_cons,
    "logcnt+cons+lowres": s_count_cons_lowres,
    "logcnt+modarea": s_count_meanarea,
}


def sweep():
    data = json.load(open(CACHE))
    gts = np.array([d["gt"] for d in data], float)

    def mae(p):
        return float(np.abs(np.array(p, float) - gts).mean())

    def rmse(p):
        return float(np.sqrt(((np.array(p, float) - gts) ** 2).mean()))

    # baselines
    def top1(groups):
        return groups[0]["count"] if groups else 0

    def oracle(groups, gt):
        return min((g["count"] for g in groups), key=lambda c: abs(c - gt)) if groups else 0

    rows = []
    rows.append(("top1", [top1(d["groups"]) for d in data]))
    for name, fn in VARIANTS.items():
        rows.append((name + " (keep)", [variant(d["groups"], fn, True) for d in data]))
        rows.append((name + " (nokeep)", [variant(d["groups"], fn, False) for d in data]))
    rows.append(("oracle", [oracle(d["groups"], d["gt"]) for d in data]))

    print(f"\n=== select scoring sweep (n={len(data)}) ===")
    print(f"{'variant':>22} {'MAE':>8} {'RMSE':>8}")
    for name, preds in rows:
        print(f"{name:>22} {mae(preds):>8.2f} {rmse(preds):>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stride", type=int, default=47)
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()
    if args.build:
        build_cache(args.stride, args.limit, args.offset)
    if args.sweep:
        sweep()


if __name__ == "__main__":
    main()
