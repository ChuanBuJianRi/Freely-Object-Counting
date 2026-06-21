# OV-CUD 训练规划：Category Head 与 Relation Head

本文档针对 `1_revised(1).md` 中定义的 OV-CUD 框架，规划其中**两个需要训练的模型头**的训练方案。

框架其余部分（SAM2、DINOv2 Region Encoder、Text Encoder、聚类 / 精修 / 计数逻辑）在第一版中全部**冻结或非训练**，因此训练目标聚焦且轻量。

---

## 0. 训练范围界定

### 0.1 需要训练的模型

| 模型头 | 角色 | 输入 | 输出 |
|---|---|---|---|
| **Category Prediction Head**（projection head + 多种评分头 + 辅助头） | 分类头 | DINOv2 区域特征 `z_i` | 类别分布 `p_i(c)` + `is_countable_i` 等辅助输出 |
| **Pairwise Relation Head** | 输出头 | 候选对特征 `phi_ij` | `A_sem[i,j]`、`A_inst[i,j]`、`A_part[i,j]` |

### 0.2 冻结 / 不训练的组件

```text
冻结（提供特征，不更新权重）:
    SAM2
    DINOv2 Region Encoder
    Text Encoder (CLIP / SigLIP)

非训练（纯算法逻辑，无参数）:
    Matrix Clustering
    Consistency Refinement
    Instance Deduplication
    Representative Selection
    Count Output

第一版不引入:
    count regressor
    density predictor
    group-level count head
    representative head
    validity head
    end-to-end joint training
```

### 0.3 总体训练策略：分阶段、解耦

```text
Stage 1: 训练 Category Head（依赖区域特征 + 文本原型）
Stage 2: 冻结 Category Head，训练 Relation Head（依赖区域特征 + 类别先验）
```

两阶段解耦的原因：
- Relation Head 的 pairwise feature 中包含 `p_i dot p_j`，依赖 Category Head 输出，先训练 Category Head 可提供稳定的类别先验。
- 解耦后单阶段调试简单，显存占用低，迭代快。

---

## 1. 共享前置：数据与候选-GT 匹配

两个 Head 共用同一套离线预处理产物，建议**一次性预计算并缓存**，避免每个 epoch 重复跑 SAM2 / DINOv2。

### 1.1 数据集要求

训练阶段允许使用、且本方法依赖的标注：

```text
object category labels
instance segmentation masks
detection boxes
region-level semantic labels (可选)
```

**禁止使用**（与方法定位冲突）：

```text
image-level count labels
density maps
count regression labels
```

推荐数据源：实例分割数据集（如 COCO / LVIS / ADE20K-instance 等），需带 instance mask + category。LVIS 的长尾类别有助于 open-vocabulary 泛化。

### 1.2 离线预计算 pipeline

对每张训练图像执行一次，缓存结果：

```text
1. SAM2 生成过完备候选 {M_i}
2. filter_obvious_noise 轻量过滤
3. build_candidate_crops -> masked / box / context crops + geometry
4. DINOv2 编码 -> z_i_mask, z_i_box, z_i_ctx
5. 候选-GT 匹配 -> 每个候选的 purity / coverage / matched_class / matched_instance_id / valid 标志
6. 缓存到磁盘 (per-image .pt / .npz)
```

> 注意：RegionFuse 若是可训练 MLP，则不要把融合后的 `z_i` 缓存死，而是缓存三路原始特征 + geometry，融合在训练 forward 中进行。第一版若 RegionFuse 也冻结/简单拼接，可直接缓存 `z_i`。

### 1.3 候选-GT 匹配（核心标签来源）

对候选 `M_i` 与 GT 实例 `G_k`：

```text
IoU_i,k      = |M_i ∩ G_k| / |M_i ∪ G_k|
purity_i,k   = |M_i ∩ G_k| / |M_i|
coverage_i,k = |M_i ∩ G_k| / |G_k|

k*                   = argmax_k IoU_i,k
purity_i             = purity_i,k*
coverage_i           = coverage_i,k*
matched_class_i      = class(G_k*)
matched_instance_id_i = id(G_k*)
```

有效性规则（决定候选是正样本 / part / ignore / background）：

```text
if max_k purity_i,k < tau_purity:
    label = background/noise（不作为类别强正样本）
elif max_k coverage_i,k < tau_part and purity_i 高:
    label = part candidate
else:
    label = valid semantic candidate
```

样本权重：

```text
w_i = purity_i * valid_i
```

低 purity 候选**绝不**硬分配 `matched_class_i` 为强正样本，否则同时污染 Category 训练和 Relation 标签。

建议超参初值：

```text
tau_purity = 0.5
tau_part   = 0.5
tau_iou_match = 0.5
```

---

## 2. Stage 1：训练 Category Prediction Head

### 2.1 结构

分类头的核心是把区域特征 `z_i` 映射到「类别分数」，但**评分方式不止 cosine 一种**。下面给出几种可选的分类头设计，可单用，也可组合（推荐 A + B 互补）。

#### 方式 A：Text-Prototype Cosine Head（开放词表主干）

```text
z_i -> projection head (MLP) -> h_i (L2 normalize)
logit_i,c = cosine(h_i, t_c) / temperature
其中 t_c = TextEncoder("a photo of a {class_name}")  # 离线固定
```

- 优点：天然支持 open-vocabulary，新增类别只需加文本原型，无需改结构。
- 缺点：受文本-视觉对齐质量限制，细粒度/长尾类别区分力弱。
- 可训练参数：projection head（+ 可学习 temperature）。

#### 方式 B：Learnable Linear / Prototype Classifier（闭集判别力强）

```text
z_i -> projection head -> h_i
logit_i,c = W_c · h_i + b_c        # 每个类别一个可学习权重向量
```

- 等价于「每类一个可学习视觉原型」，对训练集内类别判别力最强。
- 缺点：W 固定维度，**不直接支持新类别**（新类要扩 W 并微调）。
- 适合 countable 闭集类别；可与 A 拼接：训练集类用 B，未见类 fallback 到 A。

#### 方式 C：Cosine + Margin（ArcFace / CosFace 风格）

```text
logit_i,c = s · (cosine(h_i, t_c) - m · 1[c == y_i])   # 训练时对正类减 margin
```

- 在 cosine 基础上加角度间隔，显著提升类间可分性、压缩类内方差，对长尾/细粒度更稳。
- 仅训练时加 margin，推理时退化为普通 cosine，**仍兼容 open-vocabulary**。

#### 方式 D：Hybrid 双分支（推荐）

```text
score_i,c = α · cosine_head(h_i, t_c)          # 开放词表分支(A/C)
          + (1 - α) · linear_head(h_i)[c]      # 闭集判别分支(B)
```

- 闭集类用判别分支拿准确率，未见类靠文本原型分支兜底；`α` 可固定或可学习。

#### 辅助输出头（与主分类共享 h_i）

主分类之外，文档需要的几个布尔/粗粒度判断也由分类头侧输出：

```text
is_countable_i : countable vs auxiliary/fallback 二分类头（BCE）
group_type_i   : countable / auxiliary / unknown / background 粗类头（可选）
```

**默认方案**：以方式 D 为主（A 开放词表 + B 闭集判别），方式 C 的 margin 作为可选增强；外加 `is_countable` 辅助头。可训练参数：projection head + linear/prototype 权重 + 辅助头（+ 可学习 temperature / α）。冻结 Text Encoder 生成的 `t_c`。

### 2.2 损失

主分类损失（按所选分类头方式，logit 不同但形式一致）：

```text
L_cls = sum_i w_i * CE(p_i, matched_class_i)
w_i   = purity_i * valid_i
```

若用方式 C（margin），`p_i` 由带 margin 的 logit 经 softmax 得到。

辅助损失：

```text
L_count = sum_i w_i * BCE(is_countable_i, is_countable_label_i)   # countable 二分类
L_align = 可选：h_i 与正类文本原型 t_{y_i} 的对齐项（如 InfoNCE / cosine 拉近）
```

可选 InfoNCE（强化视觉-文本对齐，利于 open-vocabulary）：

```text
L_align = -sum_i w_i * log( exp(cos(h_i, t_{y_i})/τ) / sum_c exp(cos(h_i, t_c)/τ) )
```

总损失：

```text
L_category = L_cls + λ_count * L_count + λ_align * L_align
```

类别样本细化处理：

```text
low-purity candidates  -> ignore（不计入 loss）
background candidates   -> background / noise 类
part candidates         -> matched object class，权重降低；若有 part 标注则用 auxiliary part 类
```

### 2.3 词表与不平衡处理

- 词表分三类：countable / auxiliary / fallback（unknown_object, unknown_repeated_pattern, background, noise）。
- 长尾问题：对 CE 使用 class-balanced 权重或 focal loss；按类别频率做采样平衡。
- open-vocabulary 关键：训练时可**留出部分类别不参与监督**（held-out），仅在验证时加入词表，检验零样本对齐能力。

### 2.4 训练配置（建议初值）

```text
optimizer     : AdamW
lr            : 1e-3 (projection head 较小，可偏大)
weight_decay  : 1e-4
batch         : 以候选为单位，512~2048 candidates / step
temperature   : 0.07 (固定) 或可学习 (clamp 到 [0.01, 0.5])
epochs        : 20~50（数据量定）
scheduler     : cosine + warmup
```

### 2.5 Stage 1 验收指标

```text
candidate top-1 / top-k 类别准确率（仅在 valid 候选上）
held-out 类别零样本准确率
background / noise 拒识率
countable vs non-countable 二分类准确率
```

Stage 1 通过标准：valid 候选 top-1 准确率稳定，held-out 类别有非平凡准确率（验证 open-vocabulary 对齐有效）。

---

## 3. Stage 2：训练 Pairwise Relation Head

冻结 SAM2 / DINOv2 / Text Encoder / **Category Head**，只训练 Relation Head。

### 3.1 结构

输入 pairwise feature：

```text
phi_ij = [
    z_i, z_j, |z_i - z_j|, z_i * z_j,
    cosine(z_i, z_j),
    p_i dot p_j,                 # 来自已冻结的 Category Head
    IoU(mask_i, mask_j), IoU(box_i, box_j),
    center_distance_ij, scale_ratio_ij, area_ratio_ij,
    containment_i_in_j, containment_j_in_i
]
```

输出三个分支：

```text
A_sem[i,j]  : 同语义类别概率（对称）
A_inst[i,j] : 同一真实实例概率（对称）
A_part[i,j] : i 是 j 的部件的概率（有向）
```

对称分支用双向平均：`A[i,j] = (score_ij + score_ji) / 2`；有向分支保留方向。

### 3.2 标签构造

```text
y_sem_ij  = 1[valid_i and valid_j and class_i == class_j]
y_inst_ij = 1[valid_i and valid_j and instance_id_i == instance_id_j]
```

Part-whole soft target：

```text
containment_i_in_j        = |M_i ∩ M_j| / |M_i|
same_inst_ij              = 1[instance_id_i == instance_id_j]
completeness_gap_i_to_j   = max(0, coverage_j - coverage_i)

y_part_i_to_j = same_inst_ij * containment_i_in_j * completeness_gap_i_to_j
```

### 3.3 损失

```text
L_sem  = sum_ij w_i w_j * BCE(A_sem[i,j],  y_sem_ij)
L_inst = sum_ij w_i w_j * BCE(A_inst[i,j], y_inst_ij)
L_part = sum_ij w_i w_j * BCE(A_part[i,j], y_part_i_to_j)
L_relation = L_sem + L_inst + L_part
```

权重 `w_i = purity_i * valid_i`，自动抑制 background / 低质量候选对 loss 的影响。

可对三项加权 `λ_sem, λ_inst, λ_part` 调平衡（part 正样本稀疏，可适当上调或用 focal）。

### 3.4 Pair Sampling（关键）

候选对数量 O(N²)，且极度不平衡（绝大多数为 different-class 负对），必须做平衡采样：

```text
正负配比控制，每个 mini-batch 强制采样:
    same-instance pairs
    same-class different-instance pairs
    different-class pairs
    part-whole pairs
    hard negatives
    background / invalid pairs
```

Hard negatives 重点挖掘：

```text
外观相似但类别不同
同类别但不同实例
高度重叠但非同一完整实例
局部候选 vs 完整候选
```

建议：每图先在线/离线选取 top-K 候选，限制 pair 规模；用 hard negative mining（按当前模型 loss 排序）提升收敛质量。

### 3.5 训练配置（建议初值）

```text
optimizer    : AdamW
lr           : 5e-4
weight_decay : 1e-4
batch        : 以 pair 为单位，4k~16k pairs / step
pos:neg      : 每类关系约 1:3 ~ 1:5，hard neg 占负样本 30%+
epochs       : 20~40
scheduler    : cosine + warmup
loss         : BCE（part 分支可选 focal/MSE for soft target）
```

### 3.6 Stage 2 验收指标

```text
A_sem  : pair-level AUC / AP（same-category 判定）
A_inst : pair-level AUC / AP（same-instance 判定）
A_part : 方向准确率 + part-whole ranking 正确率
下游联动:
    用预测矩阵跑 clustering，检查 coarse group 纯度
    same-instance component 去重后 count 误差
```

---

## 4. 端到端联调与评测（下游闭环验证）

两个 Head 训练完成后，接入冻结的聚类 / 精修 / 计数逻辑，做整体评测（不再训练）：

```text
SAM2 -> DINOv2 -> Category Head -> Relation Head
     -> A_group clustering -> refinement
     -> same-instance dedup -> representative selection -> count
```

主指标（class-aware，与文档 §18 一致）：

```text
主指标: class-aware count MAE / RMSE / NAE / SRE
辅助: group class accuracy, top-1/top-k, unknown handling
诊断: box/mask AP, instance recall, duplicate rate, part-as-instance error rate
```

通过聚类/计数结果反向定位是 Category Head 还是 Relation Head 的瓶颈，针对性回炉。

---

## 5. 训练流程时间线（Phase 对齐）

| 阶段 | 训练内容 | 冻结项 | 产物 | 对应文档 Phase |
|---|---|---|---|---|
| 预处理 | 无（离线缓存） | SAM2/DINOv2 | 候选 + 特征 + 匹配标签缓存 | — |
| Stage 1 | Category Head | SAM2/DINOv2/Text | 稳定类别分布 | Phase 1 |
| Stage 2 | Relation Head | + Category Head | 关系矩阵 A_sem/A_inst/A_part | Phase 2 |
| 联调 | 不训练 | 全部 | coarse/refined groups | Phase 3-4 |
| 评测 | 不训练 | 全部 | class-aware count metrics | Phase 5 |

---

## 6. 实现落点（对齐代码结构）

```text
ov_cud/training/
    dataset.py        # 加载图像 + GT，触发离线缓存
    matching.py       # 候选-GT 匹配，生成 purity/coverage/valid/labels
    losses.py         # L_category, L_sem, L_inst, L_part
    train_category.py # Stage 1 入口
    train_relation.py # Stage 2 入口

ov_cud/heads/
    category_head.py  # projection head + 文本原型匹配
    relation_head.py  # pairwise 三分支输出

ov_cud/matrix/
    pairwise_features.py  # phi_ij 构造（训练与推理共用）
```

---

## 7. 关键风险与对策小结

| 风险 | 对策 |
|---|---|
| 低 purity 候选污染类别/关系标签 | `w_i = purity_i * valid_i`，ignore 规则 |
| pair 极度不平衡 | 强制配比采样 + hard negative mining |
| part-whole 方向写反 | 统一 `A_part[i,j] = P(i is part of j)`，soft target 验证 |
| 长尾 / open-vocabulary 泛化 | class-balanced loss、held-out 类别验证、文本原型对齐 |
| 重复跑 SAM2/DINOv2 拖慢训练 | 离线预计算 + 磁盘缓存 |
| 两 Head 互相影响难定位 | 分阶段解耦，Stage 2 冻结 Category Head |
