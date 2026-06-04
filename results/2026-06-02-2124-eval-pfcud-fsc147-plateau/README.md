# 2026-06-02-2124-eval-pfcud-fsc147-plateau

PF-CUD 在 FSC-147 test 全集（1190 图）上，改进 **scale-layer 计数**的尺度选层规则后的评估。

状态: **done**（GPU 全量评估完成，metrics.json 已产出；热保护安全：0 次降温，峰值 39°C）。

## 背景

PF-CUD 是 parameter-free 通用计数框架。上一轮（`2026-06-02-0144`）的全量结果中，4 个预测器：

| 预测器 | MAE | RMSE | 说明 |
|---|---|---|---|
| top1 | 63.79 | 160.07 | rank-1 组成员数（pipeline 实际输出）|
| select | 82.95 | 208.82 | group_filter 选组（最差）|
| **scale** | **47.37** | 131.91 | scale-layer 计数（最优非作弊）|
| oracle | 25.09 | 68.02 | 选最接近 GT 的组（作弊上界）|

`scale` 是最好的非作弊预测器，但其尺度选层规则 `most_stable`（选全局最平坦的内部尺度层）有系统偏差。

## 诊断（离线，全量 1190 图 sigma 直方图）

- **正确计数几乎总在某个尺度层**：`oracle_scale`（每图选最接近 GT 的尺度层 count）MAE 仅 **15.64**，远低于 group-oracle（25.09）。说明 blob 尺度空间里包含了正确答案，问题是**选层**。
- **oracle 尺度层的归一化位置随计数 regime 单调移动**（0=细，1=粗）：
  - gt 1-10：~0.89（粗尺度，少而大的物体）
  - gt 11-50：~0.82
  - gt 51-200：~0.55
  - gt 201+：~0.29（细尺度，多而小的物体）
- 因此**固定尺度层/全局最平坦点无法跨 regime 自适应**。`most_stable` 偏向粗尾，在小计数上高估（gt 1-10 时 MAE 32，均值 41 vs gt 9）。

## 改动

`pf_cud/select/scale_count.py`：尺度选层规则 `most_stable` → **`coarsest_plateau`**。

> 对相邻尺度层的**相对 count 变化** `|c_{j+1}-c_j| / mean`，把 ≤ 该曲线自身中位相对变化的层标记为 plateau 层（数据驱动的二分，非调参阈值），取其中**最粗**的一层。即"数据支持的最大稳定物体尺度"，按图自适应，无任何参数。

仍然 parameter-free（只用 median 自适应分界 + 相对变化），符合项目约束。

## 结果（GPU 全量评估，n=1190）

四个预测器最终 MAE/RMSE：

| 预测器 | MAE | RMSE | 说明 |
|---|---|---|---|
| top1 | 63.79 | 160.07 | rank-1 组成员数 |
| select | 82.95 | 208.82 | group_filter 选组（最差）|
| **scale（新 coarsest_plateau）** | **44.02** | **129.95** | 最优非作弊预测器 |
| oracle | 25.09 | 68.02 | group-oracle 作弊上界 |

scale-layer 规则 by-bucket 对比（GPU 全量，与离线一致）：

| 尺度规则 | MAE | RMSE | by-bucket MAE [1-10/11-50/51-200/201+] |
|---|---|---|---|
| most_stable（旧）| 47.37 | 131.91 | 32 / 25 / 49 / 316 |
| **coarsest_plateau（新）** | **44.02** | **129.95** | **24 / 19 / 53 / 310** |
| oracle_scale（作弊上界）| 15.64 | 66.69 | 18 / 10 / 13 / 99 |

- 整体 MAE 47.37 → **44.02**（−7%），RMSE 131.91 → 129.95。
- 小/中计数（1-50，共 741 图，占 62%）明显改善：32→24、25→19。
- 大计数（51-200）略升（49→53），201+ 仍崩溃（310，单尺度层无法兜住极端计数）。
- in-module `scale_layer_count_from_sigmas` 与离线 sweep、GPU 全量评估**三者完全一致**（44.02 / 129.95）。
- 热保护：4 卡共 236 次 poll，**0 次降温事件**，峰值温度 39°C。

## 否决的方向（已离线验证无效）

- **scale+top1 融合**：top1 与 gt 几乎不相关（corr≈0），mean/min/max/geomean 均无稳定增益。
- **consensus（用 top1 作 regime 锚 + 选最近尺度层）**：top1 太噪声，不优于 coarsest_plateau。
- 其它曲线泛函（knee/elbow/curvature/coverage-peak/median-all）均 ≥ coarsest_plateau。

## 复现

```bash
# 1. dump 全量 blob sigma 直方图（纯 CPU，8 路并行，~4 分钟）
cd bench/New_approach
for o in 0 1 2 3 4 5 6 7; do \
  /home/gaoyiyang/venvs/fsc/bin/python analysis/dump_sigma_hist.py \
    --stride 8 --offset $o --out outputs/sigma_hist/shard_$o.json & done; wait
# 2. 离线 sweep（秒级）
/home/gaoyiyang/venvs/fsc/bin/python analysis/sweep_scale_rules.py
# 3. GPU 全量评估（4 卡，~65 min/卡）
bash run_fsc147_4gpu.sh
```

## 相关文件

- 改动模块: `bench/New_approach/pf_cud/select/scale_count.py`（`_coarsest_plateau_index`）
- 离线工具: `analysis/dump_sigma_hist.py`、`analysis/sweep_scale_rules.py`
- GPU 输出: `bench/New_approach/outputs/fsc147_full_plateau/shard_offset{0..3}.json`

## notes

- 未触发热保护：4 卡 236 次 poll，0 次降温，峰值 39°C（见 `metrics.json` 的 `thermal` 块）。
- 下一步候选：把"按 regime 选层"做成显式自适应（仍需 parameter-free），目标缩小到 oracle_scale 的 15.64；或对 201+ 极端计数单独处理。
