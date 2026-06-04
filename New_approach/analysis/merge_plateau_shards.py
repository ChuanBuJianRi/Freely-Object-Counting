"""Merge the 4 plateau-rule FSC147 shards into one metrics.json over all 1190
images. Recomputes MAE/RMSE/NAE/SRE and per-gt-bucket MAE for top1/select/
scale/oracle, and aggregates the thermal block."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

OUT_DIR = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/New_approach/outputs/fsc147_full_plateau")
RESULT = Path("/home/gaoyiyang/ws_yiyang/ws_mmmu/bench/results/2026-06-02-2124-eval-pfcud-fsc147-plateau")

per = []
thermals = []
for f in sorted(glob.glob(str(OUT_DIR / "shard_offset*.json"))):
    d = json.load(open(f))
    per.extend(d["per_image"])
    thermals.append(d.get("thermal", {}))

n = len(per)
gt = np.array([r["gt"] for r in per], float)


def arr(k):
    return np.array([r[k] for r in per], float)


def mae(p):
    return float(np.abs(p - gt).mean())


def rmse(p):
    return float(np.sqrt(((p - gt) ** 2).mean()))


def nae(p):
    d = np.where(gt == 0, 1.0, gt)
    return float((np.abs(p - gt) / d).mean())


def sre(p):
    d = np.where(gt == 0, 1.0, gt)
    return float((((p - gt) ** 2) / d).mean())


def buckets(p):
    out = {}
    for label, m in {
        "1-10": (gt >= 1) & (gt <= 10),
        "11-50": (gt >= 11) & (gt <= 50),
        "51-200": (gt >= 51) & (gt <= 200),
        "201+": gt > 200,
    }.items():
        if m.any():
            out[label] = {
                "n": int(m.sum()),
                "mae": float(np.abs(p[m] - gt[m]).mean()),
                "rmse": float(np.sqrt(((p[m] - gt[m]) ** 2).mean())),
            }
    return out


metrics = {"num_images": n}
for k in ["top1", "select", "scale", "oracle"]:
    p = arr(k)
    metrics[f"{k}_mae"] = mae(p)
    metrics[f"{k}_rmse"] = rmse(p)
    if k != "oracle":
        metrics[f"{k}_nae"] = nae(p)
        metrics[f"{k}_sre"] = sre(p)
    metrics[f"{k}_by_gt_range"] = buckets(p)

# aggregate thermal: peak temp, total cooldown events/seconds, total polls
def agg(key, fn):
    vals = [t.get(key) for t in thermals if t.get(key) is not None]
    return fn(vals) if vals else None


thermal = {
    "enabled": all(t.get("enabled", False) for t in thermals) if thermals else False,
    "temp_limit_c": agg("temp_limit_c", lambda v: v[0]),
    "cooldown_sec": agg("cooldown_sec", lambda v: v[0]),
    "hysteresis_c": agg("hysteresis_c", lambda v: v[0]),
    "check_every": agg("check_every", lambda v: v[0]),
    "peak_temp_c": agg("peak_temp_c", max),
    "cooldown_events": agg("cooldown_events", sum),
    "cooldown_seconds": agg("cooldown_seconds", sum),
    "polls": agg("polls", sum),
    "num_shards": len(thermals),
}

out = {"metrics": metrics, "thermal": thermal}
json.dump(out, open(RESULT / "metrics.json", "w"), indent=2)
json.dump(per, open(RESULT / "per_image_test.json", "w"), indent=2)

print(f"merged {n} images")
print(json.dumps(metrics, indent=2))
print("\nthermal:", json.dumps(thermal, indent=2))
