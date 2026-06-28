"""计数 oracle（FSC147 dot-only）：评估"选候选"环节的计数上限。

FSC147 只有 dot 标注（每个实例一个点），无 instance mask、无类别。
本脚本用"候选 mask 是否命中 GT dot"定义两种计数 oracle，衡量候选阶段
之后理论可达的最优计数误差（MAE / RMSE）：

  Oracle-A 覆盖上界 (cover):
      pred = 被任意候选覆盖的 dot 数。
      = 每个被覆盖的 dot 都恰好数到一次的最乐观上界。
      未覆盖 dot 直接丢失，故 pred <= gt，误差全部来自候选 dot recall 缺口。

  Oracle-B 贪心一对一 (greedy set-cover):
      把每个候选映射到它覆盖的 dot 集合，贪心选最少候选覆盖所有可覆盖 dot，
      pred = 选中的候选数。
      更贴近真实"选代表候选"流程：一个候选罩住多个 dot 时只算 1，暴露欠数。

两种 oracle 都与 GT count 比，按 GT 密度区间聚合 MAE / RMSE / 平均偏差。
仅依赖缓存里的 masks_rle + 标注里的 points，不需要 DINOv2 特征。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

DEFAULT_ANN = "/home/czp/official_code/dataset/FSC147/annotation_FSC147_384.json"

BINS: List[Tuple[str, int, int]] = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, 10 ** 9),
]


def bin_of(c: int) -> str:
    for lab, lo, hi in BINS:
        if lo <= c <= hi:
            return lab
    return "100+"


def decode_masks(masks_rle: List[dict], h: int, w: int) -> List[np.ndarray]:
    out = []
    for r in masks_rle:
        m = mask_utils.decode(r).astype(bool)
        if m.shape == (h, w):
            out.append(m)
    return out


def cand_dot_sets(masks: List[np.ndarray], pts_int: List[Tuple[int, int]]) -> List[set]:
    """每个候选覆盖的 dot 索引集合。"""
    sets = []
    for m in masks:
        s = set()
        for di, (xi, yi) in enumerate(pts_int):
            if m[yi, xi]:
                s.add(di)
        sets.append(s)
    return sets


def greedy_set_cover(sets: List[set], n_dots_coverable: set) -> int:
    """贪心集合覆盖：返回覆盖所有可覆盖 dot 所需的最少候选数。"""
    remaining = set(n_dots_coverable)
    chosen = 0
    pool = [set(s) for s in sets if s]
    while remaining:
        best = max(pool, key=lambda s: len(s & remaining), default=None)
        if best is None or not (best & remaining):
            break
        remaining -= best
        chosen += 1
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="候选缓存目录（每图一个 .pt）")
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--images-file", default=None, help="可选：只评这些文件名（json 列表）")
    ap.add_argument("--out", default="", help="可选：明细 json 输出")
    args = ap.parse_args()

    ann = json.load(open(args.ann))
    if args.images_file:
        wanted = set(json.load(open(args.images_file)))
    else:
        wanted = None

    files = sorted(f for f in os.listdir(args.cache_dir) if f.endswith(".pt"))

    rows = []
    for fn in files:
        d = torch.load(os.path.join(args.cache_dir, fn), map_location="cpu", weights_only=False)
        file_name = d.get("file_name") or f"{d.get('img_id')}.jpg"
        if wanted is not None and file_name not in wanted:
            continue
        entry = ann.get(file_name)
        if entry is None or not entry.get("points"):
            continue

        h, w = int(d["height"]), int(d["width"])
        masks = decode_masks(d["masks_rle"], h, w)
        pts = np.asarray(entry["points"], dtype=np.float64)

        pts_int = []
        for x, y in pts:
            xi, yi = int(round(float(x))), int(round(float(y)))
            if 0 <= xi < w and 0 <= yi < h:
                pts_int.append((xi, yi))
        gt = len(pts_int)
        if gt == 0:
            continue

        sets = cand_dot_sets(masks, pts_int)
        covered = set().union(*sets) if sets else set()
        pred_a = len(covered)                 # 覆盖上界
        pred_b = greedy_set_cover(sets, covered)  # 贪心一对一

        rows.append({
            "file": file_name,
            "gt": gt,
            "n_cand": len(masks),
            "pred_cover": pred_a,
            "pred_greedy": pred_b,
            "dot_recall": pred_a / gt,
        })

    if not rows:
        print("没有可评估的图像")
        return

    def agg(rs, key):
        gts = np.array([r["gt"] for r in rs], float)
        preds = np.array([r[key] for r in rs], float)
        err = preds - gts
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
        }

    print(f"评估图像数: {len(rows)}  (缓存目录 {args.cache_dir})")
    print()
    print("=== 整体计数 oracle（pred - gt）===")
    for name, key in [("Oracle-A 覆盖上界", "pred_cover"), ("Oracle-B 贪心一对一", "pred_greedy")]:
        m = agg(rows, key)
        print(f"  {name:<20} MAE={m['MAE']:7.2f}  RMSE={m['RMSE']:7.2f}  bias={m['bias']:+7.2f}")

    print()
    print("=== 分 GT 区间 MAE（A=覆盖上界 / B=贪心）===")
    print(f"{'区间':<8}{'#图':>5}{'A_MAE':>9}{'A_RMSE':>9}{'B_MAE':>9}{'B_RMSE':>9}{'dotRec':>9}")
    by_bin: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_bin[bin_of(r["gt"])].append(r)
    for lab, _, _ in BINS:
        rs = by_bin.get(lab)
        if not rs:
            continue
        a = agg(rs, "pred_cover")
        b = agg(rs, "pred_greedy")
        dr = float(np.mean([r["dot_recall"] for r in rs]))
        print(f"{lab:<8}{len(rs):>5}{a['MAE']:>9.2f}{a['RMSE']:>9.2f}"
              f"{b['MAE']:>9.2f}{b['RMSE']:>9.2f}{dr:>9.3f}")

    if args.out:
        json.dump(
            {"overall": {"cover": agg(rows, "pred_cover"),
                         "greedy": agg(rows, "pred_greedy")},
             "rows": rows},
            open(args.out, "w"), indent=2, ensure_ascii=False,
        )
        print(f"\n明细写入 {args.out}")


if __name__ == "__main__":
    main()
