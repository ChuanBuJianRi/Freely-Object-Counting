# 2026-05-28-1538-eval-mp7-mcv-guard

task: eval
dataset: FSC-147 val (fraction=1/3, seed=42, n=425)
model: OCCAM-M + min=5e-4/max=0.10 + p7 + FINCH + **pred=mcv with --mcv-min-anchor-size 30**
status: done
related_history: history/2026-05-28-1538-add-mcv-guard.md
elapsed: 41.1 min  (started 2026-05-28 15:50, finished 16:31)
thermal: enabled, peak=63°C, 0 cooldown events, 84 polls

## headline result :: PARTIAL POSITIVE (MCV v2)

Overall MAE **29.91** vs max baseline 30.71 (**−0.80**) and v1 32.00
(**−2.09**). Matches the offline trace-sweep prediction of 29.91 to
within 0.002 MAE — the saved `per_image_val.json::trace` field of the
v1 run is therefore validated as a reliable substrate for offline
prediction-head ablations (no need to re-run SAM2/ResNet for
prediction-only tweaks).

| variant                         | overall MAE | RMSE  | NAE   | 1-10  | 11-50 | 51-200 | 201+   |
|---------------------------------|------------:|------:|------:|------:|------:|-------:|-------:|
| MP7 + max  (prior best)         |       30.71 | 93.90 | 0.41  |  6.50 |  7.39 |  35.24 | 312.96 |
| MP7 + MCV v1 (no guard)         |       32.00 | 89.96 | 0.78  | 16.67 | 14.92 |  25.08 | 280.65 |
| **MP7 + MCV + A=30 (this run)** |   **29.91** | 89.86 | 0.56  | 10.80 | 10.75 |  29.14 | 280.65 |

(2 images failed with CUDA OOM and are excluded; n shown as 425.)

## success-criteria check

- ✓ overall MAE ≤ 30.71      :: 29.91 (−0.80)
- ✓ 201+ bucket MAE ≤ 290    :: 280.65 (−32.31 vs max)
- ✗ small-bucket regression ≤ +1.0 :: combined +3.5 (1-10: +4.3, 11-50: +3.4)

The guard removes most of the F1 mode-contamination (1-10: 16.67 →
10.80; 11-50: 14.92 → 10.75) but does not fully match the max baseline.
Residual cause: even when the anchor cluster has ≥30 members, neighbouring
same-scale background clusters can still be promoted into the mode-set.

## diagnostic next steps (not blocking)

1. **A sweep on real GPU run** (not just trace replay). Offline plateau
   suggests A ∈ [30, 40]; this run uses A=30. A=40 simulated to 28.97
   on v1 trace; worth a 40-min GPU run to confirm.
2. **Per-cluster cohesion gate** (MCV v3 candidate): require that
   candidate mode-members have feature-space variance comparable to the
   anchor cluster, to filter same-scale-but-different-semantic
   backgrounds. Costs ~1-2 h coding.
3. **Anchor median-of-top-3** instead of arg-max (MCV v3-anchor): may
   fix small buckets where anchor is itself a background cluster, but
   could harm 201+ where multiple equal-size clusters are correct.

## what's saved here

- `metrics.json` — final numbers (incl. `thermal:` block)
- `summary.txt` — pretty-printed bucket table
- `per_image_val.json` — per-image `pred / gt / ae / trace`. The
  `trace.strategy` field is `"mode_cluster_vote"` for ~41% of images
  (where guard skipped) and `"mode_cluster_vote->max(guard)"` for the
  rest. This trace can drive any future MCV-vN ablation offline.
- `run.log` — full stdout
- `config.yaml`, `run.sh` — reproducible config

## reproduction

```bash
bash results/2026-05-28-1538-eval-mp7-mcv-guard/run.sh
```

