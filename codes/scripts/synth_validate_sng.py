"""CPU-only synthetic validation of the SNG adaptive-delta rule (SNG-method.md 7.1).

This script does NOT need SAM2, torch, or a GPU. It validates two claims that
must hold before the adaptive rule can be trusted on real FSC-147 features:

1. The closed-form eta_health(epsilon, delta, n) reproduces the eta values
   listed in SNG-method.md 6.4 (sanity check on the formula).
2. Across n in [50, 500] (the candidate-mask count range observed on FSC-147
   single + multi modes), the adaptive rule

       delta* = floor(alpha * (epsilon - 1) + (1 - alpha) * epsilon^2 / n)

   keeps clustering quality (ARI + counting MAE under the FSC-147-style
   max-cluster head) within a small constant factor of the best fixed-delta
   baseline. The point is robustness, not best-case performance.

Method:

- Generate K isotropic Gaussian clusters in d-dim feature space (default d=64,
  K=3, intra std=0.5, inter mean distance ~5). Per (n, seed, eps) we sweep
  delta in {1..eps-1} (fixed grid) plus three adaptive choices alpha in
  {0.3, 0.4, 0.5}, then record (ARI, count_mae, n_clusters_found, delta_used,
  eta).

Outputs:

- metrics.json :: dict with per-config rows + aggregate summary (best per n,
  worst-case MAE per scheme, mean MAE per scheme).
- per_run.csv  :: long-form rows for downstream plotting.
- summary.txt  :: short human-readable report.

Usage::

    python codes/scripts/synth_validate_sng.py --output-dir results/<run_id>/

Run is tagged CPU-only in the README, so the GpuGuard policy does not apply.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from occam.clustering import adaptive_delta, eta_health, sng_cluster


def make_synthetic(*, n: int, k: int, d: int, intra_std: float,
                   inter_dist: float, seed: int,
                   noise_frac: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Generate n points from k isotropic Gaussian clusters + noise points.

    - centers: random unit directions, scaled to ``inter_dist``
      (so adjacent centers are roughly inter_dist apart in d-dim space).
    - cluster sizes: as even as possible after subtracting noise.
    - noise points (label = -1): ``floor(noise_frac * n)`` points drawn
      from a wide isotropic Gaussian (std = inter_dist) — represents
      OCCAM background mask candidates that don't belong to any object class.

    Returns (features, labels) shuffled in a single permutation.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((k, d))
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    centers = base * inter_dist

    n_noise = int(math.floor(noise_frac * n))
    n_signal = n - n_noise

    sizes = [n_signal // k] * k
    for i in range(n_signal - sum(sizes)):
        sizes[i] += 1

    parts = []
    labels = []
    for cluster_id, size in enumerate(sizes):
        pts = centers[cluster_id] + intra_std * rng.standard_normal((size, d))
        parts.append(pts)
        labels.extend([cluster_id] * size)

    if n_noise > 0:
        noise = inter_dist * rng.standard_normal((n_noise, d))
        parts.append(noise)
        labels.extend([-1] * n_noise)

    X = np.concatenate(parts, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int32)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index without sklearn.

    ARI = (sum_ij C(nij,2) - [sum_i C(ai,2) * sum_j C(bj,2)] / C(n,2)) /
          (0.5 * [sum_i C(ai,2) + sum_j C(bj,2)] - [sum_i C(ai,2) * sum_j C(bj,2)] / C(n,2))
    """
    n = len(labels_true)
    if n == 0:
        return float("nan")

    true_classes = np.unique(labels_true)
    pred_classes = np.unique(labels_pred)

    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    for i, c in enumerate(true_classes):
        mask_c = labels_true == c
        for j, p in enumerate(pred_classes):
            contingency[i, j] = int(np.sum(mask_c & (labels_pred == p)))

    def comb2(x: np.ndarray | int) -> np.ndarray | int:
        return x * (x - 1) // 2

    sum_comb_c = int(comb2(contingency.sum(axis=1)).sum())
    sum_comb_k = int(comb2(contingency.sum(axis=0)).sum())
    sum_comb = int(comb2(contingency).sum())
    total_comb = comb2(n)
    if total_comb == 0:
        return float("nan")

    expected = sum_comb_c * sum_comb_k / total_comb
    max_index = 0.5 * (sum_comb_c + sum_comb_k)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_comb == expected else 0.0
    return float((sum_comb - expected) / denom)


def clusters_to_labels(clusters, n: int) -> np.ndarray:
    """Convert SNG list[Cluster] output into (n,) integer label vector."""
    out = np.full(n, -1, dtype=np.int32)
    for cid, cluster in enumerate(clusters):
        for idx in cluster.indices:
            out[idx] = cid
    return out


@dataclass
class RunRow:
    n: int
    k_true: int
    epsilon: int
    scheme: str
    delta_used: int
    eta: float
    seed: int
    ari: float
    count_max_pred: int
    count_max_true: int
    count_mae_max: int
    count_total_pred: int
    count_total_true: int
    count_mae_total: int
    n_clusters_found: int
    elapsed_sec: float


def evaluate_one(features, labels_true, *, epsilon, scheme, delta=None, alpha=0.4, seed):
    """Run one (eps, scheme) combo on a fixed (features, labels_true) instance."""
    n = len(features)
    k_true = int(np.unique(labels_true).size)
    t0 = time.time()
    if scheme.startswith("fixed"):
        clusters = sng_cluster(features, epsilon=epsilon, delta=delta)
        delta_used = int(delta)
    else:
        clusters = sng_cluster(features, epsilon=epsilon, delta=None, alpha=alpha)
        delta_used = adaptive_delta(epsilon=epsilon, n=n, alpha=alpha)
    elapsed = time.time() - t0

    pred_labels = clusters_to_labels(clusters, n)
    ari = adjusted_rand_index(labels_true, pred_labels)

    sizes_pred = sorted([len(c.indices) for c in clusters], reverse=True)
    valid = labels_true >= 0
    if valid.any():
        _, true_counts = np.unique(labels_true[valid], return_counts=True)
    else:
        true_counts = np.array([], dtype=np.int64)
    sizes_true = sorted(true_counts.tolist(), reverse=True)

    count_max_pred = sizes_pred[0] if sizes_pred else 0
    count_max_true = sizes_true[0] if sizes_true else 0
    count_total_pred = sum(sizes_pred)
    count_total_true = int(valid.sum())

    return RunRow(
        n=n,
        k_true=k_true,
        epsilon=epsilon,
        scheme=scheme,
        delta_used=delta_used,
        eta=eta_health(epsilon=epsilon, delta=delta_used, n=n),
        seed=seed,
        ari=ari,
        count_max_pred=count_max_pred,
        count_max_true=count_max_true,
        count_mae_max=abs(count_max_pred - count_max_true),
        count_total_pred=count_total_pred,
        count_total_true=count_total_true,
        count_mae_total=abs(count_total_pred - count_total_true),
        n_clusters_found=len(clusters),
        elapsed_sec=round(elapsed, 4),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True,
                   help="Directory to drop metrics.json / per_run.csv / summary.txt")
    p.add_argument("--ns", type=int, nargs="+",
                   default=[50, 100, 150, 250, 500],
                   help="Sweep over these synthetic sizes (candidate-mask counts).")
    p.add_argument("--epsilons", type=int, nargs="+", default=[10],
                   help="Sweep over these neighbourhood sizes.")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5],
                   help="Adaptive-delta blend coefficients to evaluate.")
    p.add_argument("--fixed-deltas", type=int, nargs="+",
                   default=[1, 2, 3, 5, 7],
                   help="Fixed-delta baselines to compare against.")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4],
                   help="Per-(n, eps) repeat seeds for the synthetic generator.")
    p.add_argument("--k", type=int, default=3, help="Number of true clusters.")
    p.add_argument("--d", type=int, default=64, help="Feature dimensionality.")
    p.add_argument("--intra-std", type=float, default=0.5,
                   help="Intra-cluster Gaussian std.")
    p.add_argument("--inter-dist", type=float, default=5.0,
                   help="Inter-cluster center distance.")
    return p.parse_args()


def aggregate(rows):
    """Group rows by (eps, scheme) and compute mean/worst metrics across (n, seed)."""
    by_scheme: dict[tuple[int, str], list[RunRow]] = {}
    for r in rows:
        by_scheme.setdefault((r.epsilon, r.scheme), []).append(r)
    out = []
    for (eps, scheme), group in sorted(by_scheme.items()):
        aris = [g.ari for g in group]
        mae_max = [g.count_mae_max for g in group]
        mae_tot = [g.count_mae_total for g in group]
        deltas = sorted({g.delta_used for g in group})
        out.append({
            "epsilon": eps,
            "scheme": scheme,
            "n_runs": len(group),
            "delta_values": deltas,
            "ari_mean": round(float(np.mean(aris)), 4),
            "ari_worst": round(float(np.min(aris)), 4),
            "count_mae_max_mean": round(float(np.mean(mae_max)), 3),
            "count_mae_max_worst": int(np.max(mae_max)),
            "count_mae_total_mean": round(float(np.mean(mae_tot)), 3),
            "count_mae_total_worst": int(np.max(mae_tot)),
        })
    return out


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    schemes = []
    for d in args.fixed_deltas:
        schemes.append(("fixed-{}".format(d), {"delta": d}))
    for a in args.alphas:
        schemes.append(("adaptive-a{:.2f}".format(a), {"alpha": a}))

    rows: list[RunRow] = []
    print("Running synthetic SNG validation:")
    print("  ns        =", args.ns)
    print("  epsilons  =", args.epsilons)
    print("  fixed-deltas =", args.fixed_deltas)
    print("  alphas    =", args.alphas)
    print("  seeds     =", args.seeds)
    print("  k={} d={} intra_std={} inter_dist={}".format(
        args.k, args.d, args.intra_std, args.inter_dist))

    total = len(args.ns) * len(args.epsilons) * len(args.seeds) * len(schemes)
    counter = 0
    for n, eps, seed in product(args.ns, args.epsilons, args.seeds):
        X, y = make_synthetic(n=n, k=args.k, d=args.d,
                              intra_std=args.intra_std,
                              inter_dist=args.inter_dist,
                              seed=seed)
        for name, kwargs in schemes:
            counter += 1
            row = evaluate_one(X, y, epsilon=eps, scheme=name, seed=seed, **kwargs)
            rows.append(row)
            if counter <= 5 or counter % 25 == 0 or counter == total:
                print("  [{:4d}/{:4d}] n={:3d} eps={:2d} seed={} {:20s} delta={:2d} eta={:+.2f} ARI={:+.3f} mae_max={:3d}".format(
                    counter, total, n, eps, seed, name, row.delta_used, row.eta, row.ari, row.count_mae_max))

    summary = aggregate(rows)

    # Best fixed scheme per epsilon (using mean count_mae_max)
    best_fixed_per_eps = {}
    for s in summary:
        if not s["scheme"].startswith("fixed"):
            continue
        eps = s["epsilon"]
        cur = best_fixed_per_eps.get(eps)
        if cur is None or s["count_mae_max_mean"] < cur["count_mae_max_mean"]:
            best_fixed_per_eps[eps] = s

    return rows, summary, best_fixed_per_eps, args, output_dir


def write_outputs(rows, summary, best_fixed_per_eps, args, output_dir):
    csv_path = output_dir / "per_run.csv"
    fields = list(RunRow.__annotations__.keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            w.writerow([getattr(r, k) for k in fields])

    metrics = {
        "config": {
            "ns": args.ns,
            "epsilons": args.epsilons,
            "alphas": args.alphas,
            "fixed_deltas": args.fixed_deltas,
            "seeds": args.seeds,
            "k": args.k,
            "d": args.d,
            "intra_std": args.intra_std,
            "inter_dist": args.inter_dist,
        },
        "thermal": {
            "enabled": False,
            "reason": "CPU-only synthetic validation; no GPU access required.",
        },
        "by_scheme": summary,
        "best_fixed_baseline": [
            {"epsilon": eps, **{k: v for k, v in best.items() if k != "epsilon"}}
            for eps, best in sorted(best_fixed_per_eps.items())
        ],
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    lines = []
    lines.append("=" * 70)
    lines.append("  GOC SNG adaptive-delta synthetic validation")
    lines.append("=" * 70)
    lines.append("  ns           = {}".format(args.ns))
    lines.append("  epsilons     = {}".format(args.epsilons))
    lines.append("  fixed_deltas = {}".format(args.fixed_deltas))
    lines.append("  alphas       = {}".format(args.alphas))
    lines.append("  seeds        = {}".format(args.seeds))
    lines.append("  k={} d={} intra_std={} inter_dist={}".format(
        args.k, args.d, args.intra_std, args.inter_dist))
    lines.append("")
    lines.append("Per-scheme aggregate (across all n x seed):")
    header = "  {:>3s} {:24s} {:>5s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "eps", "scheme", "n", "ARI_mean", "ARI_worst", "MAEmax_mean", "MAEmax_worst")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for s in summary:
        lines.append("  {:3d} {:24s} {:5d} {:10.4f} {:10.4f} {:11.3f} {:12d}".format(
            s["epsilon"], s["scheme"], s["n_runs"],
            s["ari_mean"], s["ari_worst"],
            s["count_mae_max_mean"], s["count_mae_max_worst"]))

    lines.append("")
    lines.append("Best fixed-delta baseline per epsilon (lowest MAEmax_mean):")
    for eps, best in sorted(best_fixed_per_eps.items()):
        lines.append("  eps={}  best={:20s}  MAEmax_mean={:.3f}  MAEmax_worst={}".format(
            eps, best["scheme"], best["count_mae_max_mean"], best["count_mae_max_worst"]))

    lines.append("")
    lines.append("Adaptive-vs-best-fixed comparison:")
    for s in summary:
        if not s["scheme"].startswith("adaptive"):
            continue
        ref = best_fixed_per_eps.get(s["epsilon"])
        if ref is None:
            continue
        gap_mean = s["count_mae_max_mean"] - ref["count_mae_max_mean"]
        gap_worst = s["count_mae_max_worst"] - ref["count_mae_max_worst"]
        ratio = s["count_mae_max_mean"] / ref["count_mae_max_mean"] if ref["count_mae_max_mean"] > 0 else float("inf")
        lines.append("  eps={}  {:20s}  Delta(mean)={:+.3f}  Delta(worst)={:+d}  ratio={:.3f}x".format(
            s["epsilon"], s["scheme"], gap_mean, gap_worst, ratio))

    lines.append("=" * 70)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    rows, summary, best_fixed, args, output_dir = main()
    write_outputs(rows, summary, best_fixed, args, output_dir)
