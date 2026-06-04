# PF-CUD 模型结构详解

> Parameter-Free Counting Unit Discovery —— 一个**无训练 / 无先验 / 无参数**的通用计数框架。
>
> 它不是端到端神经网络，而是一条**多阶段几何 / 统计推理管线**。唯一的"神经"部分是用 DINOv2 抽视觉特征，其余全部是参数自由的统计算子（Otsu / MST / rank 聚合 / MDL）。核心入口是 `PFCUDPipeline.run()`（见 `pf_cud/pipeline.py`）。

---

## 0. 数据结构（贯穿全程）

三个 dataclass（`pf_cud/data.py`）是整条管线传递的载体：

| 结构 | 含义 | 关键字段 |
|---|---|---|
| `Candidate` | 一个候选可数区域 | `mask`（布尔图）、`bbox`、`source`（"blob"/"edge"/"sam"）、`score`、`features`（visual/shape/color/spatial 四类特征字典）、`meta`（如 blob 的 `sigma`）|
| `CountGroup` | 一组候选 | `indices`（成员候选下标）、`count`、`group_type`、`score`/`confidence` |
| `CountResult` | 整图输出 | `groups`、`candidates`、`image_shape`、`meta`（含关键的 `raw_blob_sigmas`）|

---

## 管线总览

```
输入图像 (RGB)
  │
  ├─【1】候选生成 generate_candidates()
  │     ├─ BlobCandidateGenerator  (多尺度 LoG，每个候选带 meta['sigma'])
  │     ├─ EdgeCandidateGenerator  (闭合轮廓，--use_edge 开启)
  │     └─ SAMCandidateGenerator   (可选，注入 SAM 才启用)
  │     ⇒ 过完备候选集（可达数千个），并快照去重前的 blob sigma 直方图
  │
  ├─【2】去重 deduplicate_candidates()   (MST + Otsu，无 IoU 阈值)
  │
  ├─【3】特征提取 attach_features()
  │     ├─ visual  : DINOv2 ViT-S/14（缓存加载，回退 ResNet50 → Null）
  │     ├─ shape   : 形状特征
  │     ├─ color   : 颜色特征
  │     └─ spatial : 空间位置特征
  │
  ├─【4】距离融合 fused_distance()
  │     每种特征各自算 L2 距离 → rank-normalize 到 [0,1] → 等权平均
  │
  ├─【5】MST 构图 build_mst()          (最小生成树，无 kNN / 无 epsilon)
  │
  ├─【6】Otsu 割图 otsu_cut_mst()      (对 MST 边权做 Otsu，剪掉跨类长边)
  │
  ├─【7】连通分量 graph_to_groups()    (每个连通块 = 一个计数组)
  │
  ├─【8】MDL 精化 mdl_merge_refinement() (+可选 split)  (用 MDL 分数比较，无阈值)
  │
  ├─【9】假设排序 rank_groups()        (object / pattern / background 三类假设)
  │
  └─【10】输出 CountResult (groups: count + type + confidence)
```

---

## Part 1｜候选生成（Over-complete Candidate Generation）

目标：**宁可多、不可漏**，生成过完备候选，后续阶段再筛。三个来源：

### 1a. Blob 候选 `BlobCandidateGenerator`（核心，`candidates/blob_candidates.py`）

多尺度 LoG（高斯拉普拉斯）blob 检测：

1. `image_adaptive_sigmas(h,w)`：尺度**由图像尺寸推导** —— `min_sigma = short/512`，`max_sigma = short/32`，几何级数采样。这是金字塔构造规则，不是任务参数。
2. 每个 sigma 算尺度归一化 LoG 响应 `-∇²G · σ²`，堆成 `[S,H,W]`。
3. 3×3×3 邻域 `maximum_filter` 找 scale-space 局部极大。
4. **Otsu** 在所有正响应上自动定阈 `tau`，只保留 `局部极大 & 响应≥tau` 的点。
5. 每个点生成半径 `√2·σ` 的圆盘 mask，记录 `meta['sigma']` —— **这个 sigma 后面 scale 计数要用**。

### 1b. Edge 候选 `EdgeCandidateGenerator`（`--use_edge` 开启，`candidates/edge_candidates.py`）

闭合轮廓：Sobel 梯度 → **Otsu** 定 Canny 高阈值（低阈 = 0.5×高阈）→ Canny 边缘 → 形态学闭运算 + 填洞 → 取内部连通分量作为候选区域。两个阈值都来自梯度分布，非手调。

### 1c. SAM 候选 `SAMCandidateGenerator`（可选，需注入 SAM，`candidates/sam_candidates.py`）

直接用 SAM **官方默认配置**的 automatic mask generator，**不暴露任何阈值**（points_per_side、pred_iou_thresh 等都不调），过 / 欠分割交给后续 MST/Otsu/MDL 处理。

> **生成后**：管线快照一份去重前的 blob sigma 列表 `_raw_blob_sigmas`（因为去重会破坏逐尺度计数曲线），存进结果 meta。

---

## Part 2｜去重（Deduplication）`deduplicate_candidates`（`candidates/merge_candidates.py`）

多源候选大量重叠。**不用固定 IoU 阈值**，而是：

1. `overlap_iou_edges`：用 bbox 区间扫描线找**相交**的候选对（重叠是几何事实 IoU>0，不是阈值），对相交对算局部 IoU，边权 = `1−IoU`。
2. 按"是否重叠"建图，求连通分量。
3. 每个连通分量内用 **MST + Otsu cut**（`auto_partition_by_mst`）把真正重复的分到一组。
4. `choose_representative`：每个重复组选一个代表 —— 优先 mask 面积接近组中位的，其次偏好 SAM，再次 score 高者（`np.lexsort` 三级排序）。

---

## Part 3｜特征提取（Feature Extraction）

对每个候选附加四类特征，全部 L2 归一化：

| 特征 | 文件 | 内容 |
|---|---|---|
| **visual** | `features/visual.py` | 默认 **DINOv2 ViT-S/14**，取 `x_norm_clstoken`。回退链 DINOv2 → ResNet50 → Null（常量零向量）。先做 2 秒网络探测避免离线挂起。加速：把 uint8 crop 整批搬 GPU 做 Resize/CenterCrop/Normalize，batch=128 |
| **shape** | `features/shape.py` | 面积占比、长宽比、extent、紧致度 `4πA/P²`、bbox 归一宽高、7 个 Hu 矩（取对数）|
| **color** | `features/color.py` | Lab 空间的均值 / 标准差 / 三分位数，共 15 维。`rgb2lab` 整图只算一次，按 bbox 切片复用 |
| **spatial** | `features/spatial.py` | 归一化中心坐标、归一宽高、到图心距离，共 5 维 |

---

## Part 4｜距离融合（Rank-Normalized Fusion）`fused_distance`（`features/fusion.py`）

把四类特征**无权重**地融合成一个候选间距离矩阵：

1. 每类特征各自算 L2 距离矩阵（`torch.cdist` GPU 加速，等价 `scipy.pdist`）。
2. 每个矩阵做 **rank-normalize**：把上三角距离按大小排序映射到 `[0,1]` —— 用排名而非数值，避免 outlier 主导，也让不同量纲的特征可比。
3. **退化特征自动跳过**（如 NullVisual 全零，不参与稀释）。
4. 对各 rank 矩阵取**等权平均**。

---

## Part 5｜MST 构图 `build_mst`（`graph/mst.py`）

对融合距离矩阵求**最小生成树**（scipy），返回对称邻接矩阵。**无 kNN、无 epsilon** —— MST 是参数自由的图结构，自动连接最相似的候选。

---

## Part 6｜Otsu 割图 `otsu_cut_mst`（`graph/cut.py`）

对 MST 的边权做 **Otsu** 自动定阈 `tau`，**剪掉所有 > tau 的长边**（跨类边）。无 delta、无手调阈值。剪完后图自然断成若干块。

---

## Part 7｜连通分量 `graph_to_groups`（`graph/components.py`）

割完的图求连通分量，**每个连通块 = 一个初始 `CountGroup`**（count = 成员数）。

---

## Part 8｜MDL 精化（Minimum Description Length Refinement）

用最小描述长度准则做组的合并 / 拆分，**只比较 MDL 大小，无阈值**。

### 8a. MDL 评分 `score.py`

一个组的描述长度 = 各特征的 `prototype_cost`（原型复杂度）+ `gaussian_residual_code_length`（高斯残差编码，方差数据自估）+ 成员编码成本。组越紧致、越一致 → MDL 越低。

> **关键优化**：用充分统计量（成员数 `n`、特征和 `S1`、平方和 `S2`）算 MDL，合并两组只需相加统计量，把每次合并增益评估从 `O(n·d)` 降到 `O(d)`。

### 8b. 贪心合并 `mdl_merge_refinement`（默认开启）

用 **lazy-deletion 最大堆**反复选"合并增益最大"的一对组合并，直到没有正增益（`new_mdl < old_mdl` 才合）。语义等价于每步取全局最大增益的贪心。

### 8c. 拆分 `mdl_split_refinement`（`--use_mdl_split`，默认关）

对每个组内部再跑一遍 MST+Otsu 尝试拆分，若拆分后总 MDL 更低则接受。拆完再合并一次。

---

## Part 9｜假设排序 `rank_groups`（`ranking/hypothesis.py`）

给每个组算 7 个数据驱动的原始分，再 rank 聚合：

| 原始分 | 含义 |
|---|---|
| **repeatability** = `log(count)` | 出现多次加分，log 抑制巨型组 |
| **centrality** | 越靠图心越像主体 |
| **area_consistency** | 面积越一致越好 |
| **visual / shape / color_consistency** | 组内特征残差越小越好 |
| **backgroundness** | 空间铺得越开 + 重复越多 + 越不居中 → 越像背景 / 纹理 |

- 前 6 个 rank 平均 = 主体分 `main_scores`；
- `backgroundness` 最高的组被标为 `background_or_pattern`，其余标 `object_or_counting_unit`；
- 按主体分降序排列。**top1 预测器**取的就是排第一的组的成员数。

---

## Part 10｜输出 + 评估侧的多预测器

`run()` 返回 `CountResult`。评估脚本 `pf_cud/eval/eval_fsc147.py` 从同一次运行读出 4 个计数：

| 预测器 | 来源 | 说明 |
|---|---|---|
| **top1** | rank-1 组成员数 | Part 9 的输出 |
| **select** | `select_counting_groups` | 选组策略（见下）|
| **scale** | `scale_layer_count_from_sigmas` | 读尺度直方图（最优非作弊）|
| **oracle** | 选最接近 GT 的组 | 作弊上界（参考）|

### select 选组 `group_filter.py`

洞察："rank-1"太小（碎片）、"最大组"太大（巨型噪声组），**真正可数单元在中间** —— 即"同一模板重复多次"的组。

- **unit strength** = `log1p(count)`（重复尺度）与 `1/平均成对融合距离`（内部一致性 / 模板相似度）两个 rank 的平均。
- Stage 1：丢掉背景组，对 unit strength 做 **Otsu** 切分，保留高簇。
- Stage 2：按 unit strength 排序，取第一。

### scale 计数 `scale_count.py`（最近改进的部分）

**绕过整条图割管线**，直接读 Part 1 的 blob sigma 直方图：

1. 按 sigma 分箱 → 逐尺度检测数曲线。
2. `_coarsest_plateau_index`（**新规则 `coarsest_plateau`**）：算相邻层相对变化 `|c_{j+1}−c_j| / mean`，≤ 曲线自身**中位变化**的层为 plateau，取最粗那层。中位分界数据驱动、按图自适应。
3. 该层检测数即为计数。旧规则 `_most_stable_index`（选全局最平坦内部点）仍保留作对照。

> 设计依据：正确计数所在的尺度层是 **regime 相关**的（少而大的物体在粗尺度，多而小的物体在细尺度），固定 / 全局最平坦点无法跨 regime 自适应。

---

## 贯穿全局的 Parameter-free 设计

| 环节 | 用的参数自由算子 | 替代了什么手调参数 |
|---|---|---|
| blob / edge / 去重 / 割图 | **Otsu** | response / IoU / delta 阈值 |
| 构图 | **MST** | kNN 的 k、DBSCAN 的 epsilon |
| 特征融合 / 排序 / 选组 | **rank 聚合** | 手动特征权重 |
| 组精化 | **MDL 比较** | merge / split 阈值 |
| 尺度选层 | **median 自适应分界** | 固定尺度层 / 调参阈值 |
| 尺度生成 | **图像尺寸推导** | num_scales |

> 约束：CLI 不出现任何算法阈值参数（epsilon / delta / k / iou_thresh / score_thresh / num_scales / finch_thresh）。所有阈值来自数据分布。

---

## 目录结构

```
pf_cud/
  config.py            # 仅工程配置（设备 / 模型名 / 路径），无算法阈值
  data.py              # Candidate / CountGroup / CountResult
  candidates/          # blob / edge / sam / merge(dedup)
  features/            # utils / visual / shape / color / spatial / fusion
  graph/               # mst / cut / components
  mdl/                 # score / refine(merge + split)
  ranking/             # hypothesis
  select/              # group_filter / scale_count / mdl_count
  eval/                # metrics / match / eval_fsc147
  visualize/           # draw（结果可视化）
  pipeline.py          # PFCUDPipeline.run() 主入口
  run_image.py / run_dataset.py
```
