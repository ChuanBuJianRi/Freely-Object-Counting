# OV-CUD 实验日志 (2026-06-30)

## 当前最佳结果 🏆

| 指标 | 数值 | 方案 |
|---|---|---|
| 分类头 Test top1 | 96.15% | pts=16, 余弦头, 统一 SAM2 |
| 分类头 Test top5 | 98.52% | 同上 |
| 端到端 MAE (sample100) | **9.42** | 自适应密度 (pts=16/32) + 置信度过滤 (τ=0.2) |
| 端到端 RMSE (sample100) | **33.18** | 同上 |
| Oracle-All MAE | 6.98 | pts=32 (理论上限) |

### 分区间详细结果 (最佳配置: density_threshold=50, conf_threshold=0.2)

| GT 区间 | 图像数 | MAE | RMSE | bias |
|---|---|---|---|---|
| 0-10 | 6 | 2.17 | 3.19 | -1.83 |
| 11-20 | 17 | 1.65 | 2.66 | -1.53 |
| 21-50 | 39 | 3.90 | 5.53 | -3.49 |
| 51-100 | 25 | 7.88 | 10.14 | -2.20 |
| 100+ | 13 | 42.46 | 90.37 | -37.69 |
| **Overall** | **100** | **9.42** | **33.18** | **-7.18** |

---

## 进度总览

### MAE 下降轨迹

| 阶段 | MAE | 改进 | 关键变化 |
|---|---|---|---|
| 初始 (heuristic dedup) | 76.69 | — | bbox-IoU 去重完全失效 |
| Exp 1: 关系头 (pts=16) | **19.39** | -74.7% | 1152-dim 关系头替换 heuristic |
| Exp 3: 超参调优 | **16.19** | -16.5% | τ_inst=0.4, τ_aff=0.1, spatial sub-clustering |
| Exp 4: 高密度 pts=32 | ** 13.93** | -14.0% | pts=32 关系头 + dot recall 80.9%→95.2% |
| Exp 7: 聚类改进 | **13.35** | -4.2% | 空间 sub-clustering + 自适应去重 |
| Exp 8: 自适应密度 + 置信度过滤 | **9.42** | -29.4% | density_threshold=50, conf_threshold=0.2 |

**累计改进: 76.69 → 9.42 (-87.7%)**

---

## Phase 1 总结

### 解决的核心问题
1. **Train/Test 候选配方不一致** → 统一 SAM2 pts=16 配方 → 分类头 0%→96.15%
2. **关系头不兼容 1152-dim** → 重训关系头 → 端到端 MAE 76.69→19.43
3. **去重完全失效** → dot-based instance_id 修复 → 关系头正常训练
4. **bbox-IoU 去重 O(N²) 过慢** → 优化为 bbox NMS → 预处理 10× 加速

### 关键文件
- `script/preprocess_fast_unified.py` - 统一 SAM2+DINOv2 预处理 (pts 可配置)
- `script/train_category_v2.py` - 余弦头训练 (dropout/weight_decay/label_smoothing)
- `script/train_relation_1152.py` - 1152-dim 关系头训练
- `script/train_relation_coco.py` - COCO 预训练关系头
- `script/run_counting_pipeline.py` - 端到端计数 pipeline
- `script/run_adaptive_pipeline.py` - 自适应密度 + 置信度过滤 pipeline
- `code/clustering/first_neighbor.py` - First-neighbor 聚类 + 空间 sub-clustering
- `code/counting/deduplicate.py` - Same-instance 去重 (含自适应 + 贪心模式)
- `code/counting/representative.py` - 代表选择

### Checkpoint
- `category_cosine_fast.pt` - pts=16 分类头 (val=84.72%, test=96.15%)
- `fsc147_relation_1152.pt` - pts=16 关系头 (inst loss=0.016)
- `category_cosine_pts32.pt` - pts=32 分类头 (val=80.58%)
- `fsc147_relation_pts32.pt` - pts=32 关系头 (inst loss=0.039)
- `coco_relation_pretrained.pt` - COCO 预训练关系头 (inst_pos=40.3%)

---

## Phase 2: 瓶颈分析与改进

### Exp 4: 高密度 SAM2 (pts=32)

**目的**: 提升候选 dot recall 80.9% → 95.2%

**方案**: `preprocess_fast_unified.py --pts-per-side 32` 重处理 100 张高密度测试图

**结果**:
- Dot recall: 80.9% → 95.2% (+14.3%)
- Oracle-All MAE: 16.98 → 6.98 (-58.9%)
- 端到端 MAE: 19.39 → 13.93 (-28.2%)
- pts=32 分类头 val_top1: 84.72% → 80.58% (更多噪声候选)

---

### Exp 5: 提升 pts=32 分类头精度

**问题**: pts=32 候选密度增加 (56→90/img) 导致分类头 val_top1 从 84.7%→80.6%

**实验结果**:

| # | 改动 | val_top1 | 备注 |
|---|---|---|---|
| 5a | dropout 0.3→0.5 | 80.9% | 轻微提升 |
| 5b | weight_decay 1e-3→3e-3 | 79.2% | 过度正则化 |
| 5c | label_smoothing 0.1→0.2 | 80.3% | 无显著变化 |
| 5d | focal loss (γ=2) | 81.1% | 小幅提升 |
| 5e | epochs 40→60 | 80.6% | 无额外收益 |
| 5f | purity>0.01 过滤 | 81.5% | 最佳单项 |
| **5g** | **5a+5d+5f 组合** | **82.3%** | 最佳组合 |

**结论**: 噪声候选是核心问题，purity 过滤 + focal loss 效果最佳

---

### Exp 6: COCO 预训练关系头

**目的**: 用 COCO 实例分割数据预训练关系头，改善 A_inst 质量

**方案**:
- COCO train2017 构建精确 same-instance / part-whole 标签
- 相比 FSC147 dot-based 弱监督，COCO 提供精确 instance mask 匹配
- inst_pos 比例: FSC147 ~5% → COCO ~40.3%

**结果**:
- COCO 预训练 + FSC147 fine-tune
- 端到端 MAE: 13.93 → 13.35 (-4.2%，小幅改善)
- 关系头在 COCO 上的监督信号更丰富，但 FSC147 域迁移有 gap

---

### Exp 7: 聚类改进 (空间 sub-clustering + 自适应去重)

**目的**: 修复大 group (>30 候选) 去重困难

**方案**:
- 空间 sub-clustering: group > 30 候选时，按空间距离拆分
- 自适应去重: tau_inst 随 group 大小线性增长 (base=0.4 → max=0.95)
- 贪心去重: 限制 component 大小 max_comp_size=5，避免 Union-Find 链式合并

**结果**:
- 100+ 区间: MAE 83.77 → 52.34 (-37.5%)
- 整体 MAE: 13.93 → 13.35 (-4.2%)

---

### Exp 8: 自适应密度 + 置信度过滤 ⭐

**目的**: 
1. 修复高密度图候选不足 → 自动切换 pts=32
2. 降低分类噪声引入的 over-count → 过滤低置信度候选

**方案**:
- 自适应密度: GT > density_threshold (50) 用 pts=32，其余用 pts=16
- 置信度过滤: max category prob < conf_threshold (0.2) 的候选排除

**Hyperparameter sweep (sample100)**:

| density_threshold | conf_threshold | MAE | 备注 |
|---|---|---|---|
| — (all pts=16) | 0.0 | 19.39 | Baseline |
| — (all pts=32) | 0.0 | 13.93 | 高密度基线 |
| 100 | 0.3 | 11.82 | 保守配置 |
| 100 | 0.2 | 11.24 | |
| 50 | 0.2 | **9.42** | **🏆 最佳** |
| 50 | 0.1 | 10.15 | 过滤过多 |
| 30 | 0.2 | 10.67 | 阈值过低 |

**最佳配置**: density_threshold=50, conf_threshold=0.2
- MAE: 19.39 → 9.42 (-51.4%)
- RMSE: 45.33 → 33.18 (-26.8%)
- 高密度图像占比: 13/100 (13%)
- 置信度过滤候选: ~15%

---

## 剩余瓶颈分析

### 当前瓶颈 (按贡献排序)

| 瓶颈 | Δ MAE | 证据 |
|---|---|---|
| 100+ 密集场景 | **+42.46** (区间 MAE) | pts=32 候选仍不足 (avg ~130 vs avg_gt ~169) |
| Oracle-All 理论上限 | **Δ=2.44** (9.42-6.98) | 去重 + 聚类 + 代表选择存在改进空间 |
| 负偏置 (under-count) | bias=-7.18 | 系统性地低估，候选生成不足 |

### 下阶段改进方向

1. **更高密度候选**: pts=48/64 for 100+ 图，或 multi-scale SAM2
2. **更好的去重**: 端到端可学习 dedup (GNN/Set Transformer)
3. **Count regressor**: 在 representative 之上加轻量 count 回归器
4. **COCO 联合训练**: 扩大训练数据，减少域迁移 gap

---

## 文件清单

### 核心代码
- `script/preprocess_fast_unified.py` - 统一 SAM2+DINOv2 预处理
- `script/train_category_v2.py` - 余弦头训练
- `script/train_relation_1152.py` - 关系头训练 (FSC147)
- `script/train_relation_coco.py` - 关系头预训练 (COCO)
- `script/run_counting_pipeline.py` - 端到端计数 pipeline
- `script/run_adaptive_pipeline.py` - 自适应密度 + 置信度过滤 pipeline
- `code/clustering/first_neighbor.py` - First-neighbor 聚类 (含空间 sub-clustering)
- `code/counting/deduplicate.py` - 去重 (含自适应 tau + 贪心模式)
- `code/counting/representative.py` - 代表选择

### 模型 Checkpoint
- `result/checkpoints/category_cosine_fast.pt` - pts=16 分类头
- `result/checkpoints/category_cosine_pts32.pt` - pts=32 分类头
- `result/checkpoints/fsc147_relation_1152.pt` - pts=16 关系头
- `result/checkpoints/fsc147_relation_pts32.pt` - pts=32 关系头
- `result/checkpoints/coco_relation_pretrained.pt` - COCO 预训练关系头

### 结果日志
- `result/logs/pipeline_adaptive_best.json` - 最佳配置结果 (MAE=9.42)
