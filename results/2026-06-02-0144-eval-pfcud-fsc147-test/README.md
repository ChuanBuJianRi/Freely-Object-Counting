# 2026-06-02-0144-eval-pfcud-fsc147-test

PF-CUD (Parameter-Free Counting Unit Discovery) 在 FSC-147 test 全集上的首次复现尝试。

状态: **aborted**(主动中止,用于切换到方法改进方向;非失败)。

## 这次 run 做了什么

- 复现 `bench/New_approach` 的 PF-CUD 想法:blob 候选 → 去重(MST+Otsu) → 特征(visual DINOv2 + shape + color + spatial) → rank-fusion → MST → Otsu cut → connected components → MDL merge → hypothesis ranking。
- 4 卡并行(GPU 4/5/6/7),`images[k::4]` 分片,接入项目规范的 `GpuGuard` 热保护。
- 中止时进度约 65/1190(~5.5%),未产出最终 metrics。分片日志保留在 `run_shard{0..3}.log`。

## 环境(重要,下次直接复用)

- venv: `/home/gaoyiyang/venvs/fsc`(Python 3.11)
- torch: **2.6.0+cu124**(本机 NVIDIA 驱动 12080/CUDA12.8;最初误装 cu13 版导致 `cuda.is_available()=False`,必须用 cu124)
- 数据集: `/home/gaoyiyang/ws_yiyang/datasets/FSC147`(已下全,6146 图)
- 原 `eval_fsc147.py` 默认数据集路径是另一台机的 `/home/czp/...`,已改为本机路径并接入 `GpuGuard`、新增 `--offset`(多卡分片,工程参数)。

## 关键技术发现

### 1. 性能瓶颈不在 GPU,在 CPU 的 color 特征(已修复)

profiling 单图(2382 候选):color **23.8s** / mdl_merge 8.6s / shape 7.4s / DINOv2(GPU)5.1s。
根因: `color_feature` 对每个候选都 `rgb2lab(整图)` 一次。
修复: `pf_cud/features/color.py` 向量化(整图 Lab 只算一次,每候选只在 bbox 内取 masked 像素统计)。
效果: color **23.8s→0.18s**(×130),单图 ~36s→~4.5s,数值完全一致(diff=0)。

### 2. 核心方法问题: top1 恒为很小的数(ranking 失败,grouping 是好的)

- `top1`(rank-1 组成员数,方法实际预测)系统性偏小(2~4);`oracle`(最接近 GT 的组)MAE 很低(30 图冒烟 ~9.8)。
- 结论: **正确的可数组确实已被生成出来,问题在 hypothesis ranking 没把它排第一**。
- 原因: `rank_groups` 的 6 个信号里 4 个是 consistency(visual/shape/color/area),小组(2~3 成员)组内 residual≈0 → consistency 满分 → 系统性偏爱小紧凑组。

### 3. 用户的方法诉求(下一步主线)

> 不要 policy / 不要调参 / 不要堆策略,要**自适应**地找到"该数哪个类别/数字"。

- 据此**否决**了 largest / total / repeat-only / area-dominant / count-Otsu 这类"换一个固定选组规则"的方案(它们本质都是 policy)。
- 尝试了**自适应 MDL 计数选择**(新模块 `pf_cud/select/mdl_count.py`):用 MST 边扫描得到一族分割,对每个分割算总描述长度 L(k),argmin 选 K*,再取"压缩增益最大"的单元作为计数。**理念对,但当前实现失败**:

  3 图冒烟(no_visual): mdl MAE=362(gt=8 时 mdl=797),远差于 top1。
  失败原因:
  - blob 候选 2000+,大量是 LoG 在纹理/背景上的噪声响应;
  - "压缩增益最大"判据 ≈ 成员数 × 单位压缩,被**噪声大簇**主导 → 退化成 largest 的变种;
  - Phase 1(无 visual)特征区分力弱,噪声 blob 与真实物体在 shape/color/spatial 上分不开。

## 下一步 TODO(按优先级)

1. **给 MDL 选组加 visual(DINOv2)**:噪声 blob 的 DINO 特征应与真实物体差异大,有望让 MDL 把噪声单独编码(低增益)、真实重复物体高增益。最小改动,最可能见效。验证脚本: `analysis/compare_selection.py`(去掉 `--no_visual`)。
2. 若仍不行: MDL 增加"背景/离群"独立编码桶,避免噪声塞进可数单元;或把可数性判据从"总增益"改成"每实例增益 × 规整度",抑制靠数量堆出来的大噪声簇。
3. 抑制 blob 噪声候选(仍需 parameter-free,如基于响应分布的自适应剔除)。
4. 方案验证后,用优化代码(color 已加速)重跑全量 test,产出最终 metrics.json(含 thermal 块)。

## 相关文件

- 评估器(已改): `bench/New_approach/pf_cud/eval/eval_fsc147.py`
- 加速(已改): `bench/New_approach/pf_cud/features/color.py`
- 新方法模块: `bench/New_approach/pf_cud/select/mdl_count.py`
- 对照实验: `bench/New_approach/analysis/compare_selection.py`、`analysis/diagnose_selection.py`
- 启动/合并: 本目录 `run_4gpu.sh`、`merge_shards.py`、`config.yaml`
