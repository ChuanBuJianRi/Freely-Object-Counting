"""诊断 1：SAM candidate dot recall（FSC147）

FSC147 有 dot annotations，可逐点检查每个 GT dot 是否被至少一个 SAM candidate 覆盖：

    dot_recall = num_dots_covered_by_any_candidate / num_gt_dots

按 GT count 区间（0-10 / 11-20 / 21-50 / 51-100 / 100+）聚合，定位
"候选阶段就丢点"的密度区间——若 51-100 区间 recall 已低，则分类头再训也救不了。

数据来源
--------
- 候选：预计算缓存 .pt（每张图一个），字段 masks_rle(list[COCO RLE]) + height/width + gt_count。
- GT dot：dataset/FSC147/annotation_FSC147_384.json 的 points 字段。
  经验证 points 已是 384 分辨率图像坐标（与缓存 height/width 一致），无需再乘 ratio_h/ratio_w。
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
DEFAULT_CACHES = [
    "/home/czp/ws_yiyang/ovcud_cache/fsc147_test",
    "/home/czp/ws_yiyang/ovcud_cache/fsc147_train",
]

# (label, lo, hi)：闭区间 [lo, hi]，hi=None 表示无上界
BINS: List[Tuple[str, int, int]] = [
    ("0-10", 0, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("100+", 101, None),
]


def bin_of(count: int) -> str:
    for label, lo, hi in BINS:
        if count >= lo and (hi is None or count <= hi):
            return label
    return "100+"


def decode_candidate_union(masks_rle: List[dict], h: int, w: int) -> np.ndarray:
    """把所有候选 mask 解码并 OR 成一张 (H,W) bool 覆盖图。

    dot recall 只关心"是否被任意候选覆盖"，所以并集就足够，且比逐 mask 索引快。
    """
    if not masks_rle:
        return np.zeros((h, w), dtype=bool)
    union = np.zeros((h, w), dtype=bool)
    for r in masks_rle:
        m = mask_utils.decode(r).astype(bool)
        if m.shape != (h, w):
            # 理论上不会发生；保险起见跳过尺寸不符的候选
            continue
        union |= m
    return union


def dilate_bool(arr: np.ndarray, radius: int) -> np.ndarray:
    """对 bool 覆盖图做半径 radius 的方形膨胀（容忍 dot 落在 mask 边缘几像素外）。"""
    if radius <= 0:
        return arr
    out = arr.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(arr)
            ys = slice(max(0, dy), arr.shape[0] + min(0, dy))
            xs = slice(max(0, dx), arr.shape[1] + min(0, dx))
            sy = slice(max(0, -dy), arr.shape[0] + min(0, -dy))
            sx = slice(max(0, -dx), arr.shape[1] + min(0, -dx))
            shifted[ys, xs] = arr[sy, sx]
            out |= shifted
    return out


def image_dot_recall(
    cache_pt: dict,
    points: np.ndarray,
    radius: int,
) -> Tuple[int, int]:
    """返回 (num_covered, num_dots)。points 为 (N,2) 的 (x,y) 384 坐标。"""
    h, w = int(cache_pt["height"]), int(cache_pt["width"])
    union = decode_candidate_union(cache_pt["masks_rle"], h, w)
    union = dilate_bool(union, radius)

    covered = 0
    n = 0
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            # dot 落在图像外（极少数标注/缩放误差），不计入分母
            continue
        n += 1
        if union[yi, xi]:
            covered += 1
    return covered, n


def main() -> None:
    ap = argparse.ArgumentParser(description="FSC147 SAM candidate dot recall 诊断")
    ap.add_argument("--ann", default=DEFAULT_ANN, help="FSC147 annotation json")
    ap.add_argument(
        "--cache",
        nargs="+",
        default=DEFAULT_CACHES,
        help="一个或多个预计算缓存目录（每个 .pt 一张图）",
    )
    ap.add_argument(
        "--radius",
        type=int,
        default=0,
        help="dot 命中容差半径（像素，方形膨胀）；0=严格落点命中",
    )
    ap.add_argument("--limit", type=int, default=-1, help="每个缓存最多处理多少张（调试用）")
    ap.add_argument("--out", default="", help="可选：把逐图明细写出为 jsonl")
    args = ap.parse_args()

    ann = json.load(open(args.ann))

    # 区间聚合器：micro（按 dot 累加）+ macro（按图平均）
    bin_covered: Dict[str, int] = defaultdict(int)
    bin_dots: Dict[str, int] = defaultdict(int)
    bin_imgs: Dict[str, int] = defaultdict(int)
    bin_img_recall_sum: Dict[str, float] = defaultdict(float)

    detail_fp = open(args.out, "w") if args.out else None
    n_done = 0
    n_skip_noann = 0

    for cache_dir in args.cache:
        if not os.path.isdir(cache_dir):
            print(f"[warn] 缓存目录不存在，跳过: {cache_dir}")
            continue
        files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        if args.limit > 0:
            files = files[: args.limit]
        for fn in files:
            d = torch.load(os.path.join(cache_dir, fn), map_location="cpu", weights_only=False)
            file_name = d.get("file_name") or f"{d.get('img_id')}.jpg"
            entry = ann.get(file_name)
            if entry is None or not entry.get("points"):
                n_skip_noann += 1
                continue
            points = np.asarray(entry["points"], dtype=np.float64)
            gt_count = int(d.get("gt_count", len(points)))

            covered, n_dots = image_dot_recall(d, points, args.radius)
            if n_dots == 0:
                continue

            label = bin_of(gt_count)
            bin_covered[label] += covered
            bin_dots[label] += n_dots
            bin_imgs[label] += 1
            bin_img_recall_sum[label] += covered / n_dots

            if detail_fp is not None:
                detail_fp.write(
                    json.dumps(
                        {
                            "file_name": file_name,
                            "gt_count": gt_count,
                            "bin": label,
                            "covered": covered,
                            "n_dots": n_dots,
                            "recall": round(covered / n_dots, 4),
                        }
                    )
                    + "\n"
                )
            n_done += 1
            if n_done % 200 == 0:
                print(f"  ...processed {n_done} images")

    if detail_fp is not None:
        detail_fp.close()

    # ---- 汇总输出 ----
    print()
    print(f"处理图像数: {n_done}  (无 dot 标注跳过: {n_skip_noann})  容差半径: {args.radius}px")
    print(f"命中半径说明: dot 落在任意候选 mask{'（含 %dpx 膨胀）' % args.radius if args.radius else ''}内即算覆盖")
    print()
    header = f"{'GT 区间':<10}{'#图':>6}{'#dots':>10}{'micro recall':>16}{'macro recall':>16}"
    print(header)
    print("-" * len(header))

    tot_cov = tot_dot = tot_img = 0
    tot_macro = 0.0
    for label, _, _ in BINS:
        imgs = bin_imgs[label]
        dots = bin_dots[label]
        cov = bin_covered[label]
        if imgs == 0:
            print(f"{label:<10}{0:>6}{0:>10}{'-':>16}{'-':>16}")
            continue
        micro = cov / dots if dots else 0.0
        macro = bin_img_recall_sum[label] / imgs
        print(f"{label:<10}{imgs:>6}{dots:>10}{micro:>16.4f}{macro:>16.4f}")
        tot_cov += cov
        tot_dot += dots
        tot_img += imgs
        tot_macro += bin_img_recall_sum[label]

    print("-" * len(header))
    if tot_dot:
        print(
            f"{'ALL':<10}{tot_img:>6}{tot_dot:>10}"
            f"{tot_cov / tot_dot:>16.4f}{tot_macro / tot_img:>16.4f}"
        )
    print()
    print("micro = 总命中 dots / 总 dots（被密集大图主导）")
    print("macro = 每张图 recall 取均值（每张图等权）")


if __name__ == "__main__":
    main()
