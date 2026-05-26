# 项目背景 & 第一轮 Deep Research 请求

## 1. 项目（GOC - Freely Object Counting）

- **目标**：在开放世界场景下做 **training-free / label-free / class-free** 的 object counting。给定任意图像 + 任意类别（文本 prompt 或 reference patch），输出该类别物体数量，**不做** 任务相关训练，**不用** 数据集 GT，**不限** 类别表。
- **基线**：OCCAM（Class-Agnostic, Training-Free, Prior-Free and Multi-Class Object Counting；Spanakis et al., arXiv 2601.13871）。我已有 PDF。
- **路线**：以 OCCAM 为流水线骨架，**逐模块替换/增强**，在保持 training-free / label-free / class-free 的前提下提升 counting 精度与鲁棒性。

### OCCAM pipeline 摘要

1. **Stage 1 — Proposals**：class-agnostic mask/region proposals（SAM 家族）。
2. **Stage 2 — Features**：对每个 proposal 提取视觉/视觉-语言特征（CLIP / DINO 等冻结 backbone）。
3. **Stage 3 — Matching**：用 text prompt 或 exemplar patch 在特征空间匹配目标类。
4. **Stage 4 — Aggregation**：相似度阈值/聚类选出目标 proposals → 最终 count（可扩展 multi-class）。

### 可改造的轴（不预设答案，需要你帮忙评估）

- proposal generator：SAM 之外的替代或后处理（mask 粒度、NMS、尺度、密度感知采样）。
- feature backbone：CLIP / DINOv2 / EVA-CLIP / SigLIP 单/集成；frozen vs. light adapter。
- matching / scoring：更好的相似度函数、calibration、text+exemplar 联合 prototype、hard-negative。
- aggregation：从噪声 proposals 中稳健计数（聚类 / density estimation / 冗余去除）。
- multi-class：共现类别消歧。
- efficiency：每个模块的 latency / memory profile。

### 硬约束

- training-free（除非显式批准，不在 counting 数据集上做 fine-tune / gradient）。
- label-free（inference 不用 GT）。
- class-free（无固定类别表；查询是 text prompt 或 exemplar）。
- 每个实验 run 落到 `results/<run_id>/`，含 `config.yaml` / `metrics.json` / `log.txt`，可复现。

### 项目当前状态

- 仓库骨架已搭好（`codes/ library/ results/ history/ memory/ skills/ tmp/`），每个目录有 agent 可读的 `index.md`。
- 安装/编写了若干 skills：counting-eval, paper-reading, ablation-runner, pseudo-label-pipeline, experiment-logger, figure-maker, cs-research-workflow, academic-researcher, research, content-research-writer。
- **尚未** 跑通 OCCAM 基线，**尚未** 落地任何 ablation。
- 候选 benchmark：FSC-147, CARPK，以及其它 open-world counting 数据集。

---

## 2. 我要你（Deep Research）回答的问题

请基于上面背景，输出一份结构化的 deep research 报告，覆盖以下几块。**重要**：每个论断都要给一手出处（论文 arXiv 链接、官方 repo、benchmark 主页、leaderboard），不要泛泛而谈。

### A. SOTA 全景（重点）

1. **2024–2026** 与 GOC 设定（training-free OR label-free OR class-free / open-vocabulary object counting）相关的论文清单，覆盖：
   - reference-less / zero-shot counting（CounTX, CLIP-Count, ZSC, GroundingREC, T-Rex2, DAVE, GeCo, OmniCount, CountGD 等及更新工作）。
   - training-free counting / open-world counting / VLM-based counting。
   - 与 OCCAM 思路最接近 / 最有竞争性的工作（重点对比）。
   - 给出每篇：方法一句话总结、是否真正 training-free、所用 backbone、FSC-147 / CARPK / 其它 benchmark 上的 MAE/RMSE、与 OCCAM 的相对差距（如果有公开数）。
2. 输出一张**对比表**（markdown）：method × {年份, training-free?, label-free?, class-free?, proposal/backbone/matching/aggregation 模块, FSC-147 val/test MAE, CARPK MAE, link}。

### B. OCCAM 内部模块的改进机会（优先级排序）

针对 OCCAM 的四个 stage，结合 A 节调研，给出：
- 每个 stage 当前已知的弱点（来自 OCCAM 论文自述 + 后续工作 critique）。
- 在 training-free 约束下，**最值得首先尝试** 的替换方案（每个 stage 给 2–3 个候选，按预期收益/实施成本排序），并说明理由 + 引用支持。
- 哪些改造**会破坏** training-free / label-free / class-free 中的某条（明确标红）。

### C. Benchmark & 评测建议

- FSC-147, CARPK, 以及 2024–2026 出现的新 open-world counting benchmark（如 OmniCount-191, 等）的对比：规模、类别数、标注形式（点 / 框 / mask）、是否适合 class-free 评测。
- 推荐我**首轮** 跑哪 1–2 个 benchmark 复现 OCCAM，理由是什么（社区可比性、复现难度、数据获取）。
- 评测指标的坑：MAE/RMSE 的常见报告差异，是否要报 NAE、SRE、per-class breakdown。

### D. 第一阶段（接下来 2–4 周）的具体行动建议

- 复现 OCCAM 的最小路径（哪个开源实现可参考？是否有官方 repo？如果无，最接近的可复用代码是什么？）。
- 第一个 ablation 该从哪个 stage 切入，理由（结合 B 节优先级）。
- 风险与未知（数据访问、显存/算力、SAM/CLIP 版本兼容性 等）。

### E. 参考文献

完整 references list，包含每个引用的标题 + 作者 + 年份 + arXiv 或官方链接。

---

## 3. 输出格式要求

- 中文为主，专有名词与方法名保留英文。
- 用 markdown，含上述 A/B/C/D/E 五节，每节有清晰标题。
- 对比表必须是真正的 markdown 表，不要用图片或散文代替。
- 不确定的数字（如某方法在某 benchmark 上的 MAE）必须标注 "未在原文找到" 而不是编造。
- 末尾给一段 200 字以内的 **executive summary**，写在最前面（TL;DR）。
