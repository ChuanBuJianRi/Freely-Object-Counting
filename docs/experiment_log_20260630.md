# OV-CUD 实验日志 (2026-06-30)

## 当前最佳结果

| 指标 | 数值 | 方案 |
|---|---|---|
| 分类头 Test top1 | 96.15% | pts=16, 余弦头, 统一 SAM2 |
| 分类头 Test top5 | 98.52% | 同上 |
| 端到端 MAE (sample100) | **19.39** | pts=16 + 1152-dim关系头, τ_inst=0.4, τ_aff=0.1 |
| Oracle-All MAE | 6.98 | pts=32 (理论上限) |

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
- `script/run_counting_pipeline.py` - 端到端计数 pipeline
- `code/clustering/first_neighbor.py` - First-neighbor 聚类
- `code/counting/deduplicate.py` - Same-instance 去重
- `code/counting/representative.py` - 代表选择

### Checkpoint
- `category_cosine_fast.pt` - pts=16 分类头 (val=84.72%, test=96.15%)
- `fsc147_relation_1152.pt` - pts=16 关系头 (inst loss=0.016)
- `category_cosine_pts32.pt` - pts=32 分类头 (val=80.58%)
- `fsc147_relation_pts32.pt` - pts=32 关系头 (inst loss=0.039)

## Phase 2: 提升 pts=32 分类头精度

### 问题定位
pts=32 候选密度增加 (56→90/img) 导致分类头 val_top1 从 84.7%→80.6%。
原因：更多候选包含噪声/局部/模糊视图，训练时过拟合到噪声。

### Exp 5 实验矩阵

| # | 改动 | 预期 |
|---|---|---|
| 5a | dropout 0.3→0.5 | 减少过拟合 |
| 5b | weight_decay 1e-3→3e-3 | 更强正则化 |
| 5c | label_smoothing 0.1→0.2 | 软化标签 |
| 5d | focal loss (γ=2) | 关注难样本 |
| 5e | 训练更长 (40→60 epochs) | 充分收敛 |
| 5f | purity-based 样本过滤 (purity>0.01) | 去除噪声候选 |
| 5g | 组合: 5a+5b+5c+5e | 综合正则化 |
