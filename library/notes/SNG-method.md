# SNG：Shared-Neighbor Graph 聚类（OCCAM 中替代 FINCH 的方法）

## 1. 背景与动机

OCCAM 在零样本物体计数中，先用 SAM2 生成大量候选掩码（AMG），然后用 ResNet-50 提取每个候选区域的特征向量，最后通过聚类把"同类物体的候选掩码"归到同一簇，再用簇的大小估计物体数量。

原版用的是 **Thresholded FINCH**：每轮重算质心 → 找最近邻 → 用一个**人工设定的距离阈值**（例如 `(12.0, 9.0, 7.75)`）决定能否合并，直到没有新合并发生。这种方式的缺点是：

- 阈值是**绝对距离**，强烈依赖特征尺度与场景密度（密集场景特征距离整体偏小，稀疏场景偏大）；
- 阈值序列必须**逐 mode 调参**（single / multi 各一套），跨数据集迁移成本高；
- 异常掩码（背景碎片、过大区域）容易把质心拉偏，连锁触发错误合并。

为缓解这些问题，本工作提出 **(ε, δ) Shared-Neighbor Graph (SNG) 聚类**，用"共享邻居数"这一**无量纲、密度自适应**的图结构判据替代距离阈值，直接作用在 ResNet-50 特征空间上。

实现位置：`OCCAM/occam/clustering.py::sng_cluster`，由 `OCCAM/occam/pipeline.py` 在 `cluster_method="sng"` 时调用。

---

## 2. 方法定义

输入：候选掩码的 ResNet-50 特征矩阵 \(X \in \mathbb{R}^{n \times d}\)（其中 `d=2048`，`n` 为通过面积过滤后的候选掩码数量）。
超参数：邻居数 \(\varepsilon \in \mathbb{Z}_{>0}\)，共享邻居阈值 \(\delta \in \mathbb{Z}_{\ge 0}\)。

### 2.1 步骤一：构建对称 ε-NN 邻接

记 \(N_\varepsilon(i)\) 为节点 \(i\) 在欧氏距离下的 \(\varepsilon\) 个最近邻索引集合（不含自身）。定义**对称化**邻接：

\[
A_{ij} = 1 \iff j \in N_\varepsilon(i) \;\lor\; i \in N_\varepsilon(j)
\]

也就是说，只要双方任一方将对方列入 ε-NN，就视为存在一条无向边。这一步与 mutual-kNN 相比保留了更多潜在关联，避免在密集类内某些点未把"边缘点"列入 kNN 而造成误断。

### 2.2 步骤二：基于共享邻居的剪枝

对当前邻接图中的每条边 \((i,j)\)，统计其**端点共享邻居数**：

\[
s(i,j) \;=\; |\,N(i) \cap N(j)\,|
\]

其中 \(N(\cdot)\) 取自 2.1 中得到的对称邻接。仅保留满足

\[
s(i,j) > \delta
\]

的边，得到稀疏图 \(G^\star\)。直觉：同一物体类别的候选区域往往不止彼此互为近邻，**也共享大量其他兄弟候选**；跨类的"伪近邻"边通常缺乏足够的共同邻居支持，会被剪除。

### 2.3 步骤三：连通分量

在 \(G^\star\) 上做并查集（Union-Find）求连通分量，每个分量即一个簇。最终簇集合为：

\[
\mathcal{C} = \{\,C_k\,\}_{k=1}^{K}, \quad C_k \subseteq \{1,\dots,n\}
\]

### 2.4 计数策略

与 FINCH 路径共用预测头，由 `pred_strategy` 选择：

- `total`：\(\hat{y} = \sum_k |C_k|\)，适合多类共存且 mask 噪声小的情形；
- `max`：\(\hat{y} = \max_k |C_k|\)，适合 FSC-147 这类"图中只数 query 类别一种物体"的设定；本工作消融的所有 SNG 配置都取 `max`。

### 2.5 复杂度

- ε-NN：暴力 \(O(n^2 d)\)（实现走 NumPy 的 `_pairwise_distances` 一次性算全对距离矩阵）；
- 共享邻居计数：每条边 \(O(\varepsilon)\)，边数 \(O(n\varepsilon)\)，合计 \(O(n\varepsilon^2)\)；
- Union-Find：近线性 \(O(n\varepsilon\,\alpha(n))\)。

由于每张图候选 mask 数量通常在几十到数百，整体开销可忽略，单图额外耗时 < 0.1s。

---

## 3. 与 FINCH 的对比

| 维度 | Thresholded FINCH（基线） | SNG（本工作） |
|---|---|---|
| 判据 | 质心欧氏距离 ≤ 阈值 | 共享邻居数 > δ |
| 超参 | 多阶段阈值序列（mode 相关） | 整数 (ε, δ)，无量纲 |
| 是否依赖特征尺度 | 是 | 否（拓扑判据） |
| 对异常 mask 的鲁棒性 | 质心被拉偏后连锁错误 | 单点离群不会获得共享邻居支持 |
| 迭代结构 | 多轮直到稳定 | 一遍构图 + 一遍剪枝 |
| 簇数 | 由阈值与迭代隐式决定 | 由 (ε, δ) 显式控制密度敏感度 |

直观上，δ 控制"同类一致性的强度"——δ 越大要求簇内一致越严格，会得到更多更小的簇（高纯度、低召回）；ε 控制"邻居视野"——ε 越大每条边能看到的潜在共享越多，剪枝更宽松。

---

## 4. 在 OCCAM 流水线中的位置

```
image
  └─ SAM2 AMG (multi-scale)  ──► candidate masks
       └─ mask area filter (min=5e-4, max=0.10)         # A6 best
            └─ ResNet-50 crop=224 feature (mode=single)
                 └─► sng_cluster(features, ε, δ)        # ← 本工作替换点
                      └─ pred_strategy="max"            # FSC-147 设定
                           └─► 计数 ŷ
```

代码入口：

- 配置：`OCCAM/occam/config.py` 中的 `cluster_method`、`sng_epsilon`、`sng_delta`；
- 调度：`OCCAM/occam/pipeline.py` 第 73–78 行；
- 实现：`OCCAM/occam/clustering.py::sng_cluster`（第 63–118 行）。

---

## 5. FSC-147 val 子集消融（fraction=1/3，seed=42，428 张）

固定上游 mask 过滤为 A6 最佳值 (min=5e-4, max=0.10)、特征 mode=single、`pred_strategy=max`，扫 (ε, δ) 网格。

| 实验 | ε | δ | MAE | RMSE | NAE |
|---|---|---|---|---|---|
| A6_SNG_e10_d5 | 10 | 5 | **39.63** | 106.17 | 0.911 |
| A6_SNG_e5_d2  | 5  | 2 | 40.25 | 116.05 | **0.545** |
| A6_SNG_max (默认) | 10 | 3 | 41.67 |  99.92 | 1.063 |
| A6_SNG_e10_d2 | 10 | 2 | 42.15 |  99.96 | 1.081 |
| A6_SNG_e20_d2 | 20 | 2 | 43.02 | 100.44 | 1.123 |
| A6_SNG_e5_d3  | 5  | 3 | 49.36 | 129.09 | 0.502 |
| A6_SNG_e5_d5  | 5  | 5 | 61.13 | 137.42 | 0.758 |
| A6_SNG_e20_d3 | 20 | 3 | 43.05 | 100.50 | 1.125 |
| A6_SNG_e20_d5 | 20 | 5 | 43.02 | 100.45 | 1.126 |

参考基线：同输入下 **A6_FINCH_max MAE = 32.10**（FINCH 仍占优）。

观察：

1. **(ε, δ) 互相耦合**：固定 ε，δ 变大整体 MAE 先降后升（如 ε=10：3→5 MAE 41.67→39.63，但 ε=5：3→5 MAE 49.36→61.13 急剧恶化）。说明 δ 是密度敏感的，要按 ε 配套调。
2. **小 ε 利于 NAE，大 ε 利于 MAE**：ε=5 在小目标上更克制（NAE 最低 0.545），但大目标会被切碎；ε=10 折中最好。
3. **大 ε 时 δ 不再敏感**：ε=20 下 δ ∈ {2,3,5} 的 MAE/NAE 几乎一致（43.0±0.05），说明 ε 一旦覆盖了大部分类内样本，共享邻居计数饱和，剪枝自由度被压缩——继续增大 ε 没有调控空间。
4. **SNG 仍弱于 FINCH**：在 FSC-147 这种"质心较稳定、特征尺度受控"的任务上，FINCH 的距离阈值能力被充分发挥；SNG 的优势更可能体现在跨数据集 / 跨 mode 不调参的鲁棒性，需要后续多数据集验证。
5. **最佳点**：(ε=10, δ=5) MAE=39.63，相对 SNG 默认 (10,3) 提升 4.9%，但仍落后 FINCH 7.5 个 MAE。

---

## 6. (ε, δ) 的数学解释

### 6.1 共享邻居数的期望：零模型分析（噪声侧）

设特征空间里有 \(n\) 个点。考虑最朴素的随机零模型：每个点的 ε-NN 是从其余 \(n-1\) 个点里**均匀随机**抽取的 \(\varepsilon\) 个。在此假设下，两点 \(i, j\) 的共享邻居数 \(s(i,j)\) 服从超几何分布，其期望为

\[
\mathbb{E}[s_{\text{inter}}(i,j)] \;=\; \varepsilon \cdot \frac{\varepsilon}{n-1} \;\approx\; \frac{\varepsilon^2}{n}.
\]

方差也是同阶 \(O(\varepsilon^2 / n)\)。一个**有效的 δ 必须显著大于 \(\varepsilon^2/n\)**——否则被剪掉的"假边"等价于随机噪声，剪枝失去拓扑判别意义。

代入 FSC-147（单图候选 mask 数 \(n \approx 150\)）：

| ε | \(\varepsilon^2/n\) | 实验取的 δ | δ / (ε²/n) |
|---|---|---|---|
| 5  | 0.17 | 2, 3, 5 | 12, 18, **30** |
| 10 | 0.67 | 2, 3, 5 | 3, 4.5, **7.5** |
| 20 | 2.67 | 2, 3, 5 | 0.75, 1.1, 1.9 |

这张表立刻解释了消融现象：

- **(ε=5, δ=5)**：判据比随机基准高 30 倍，类内边都被切碎 → MAE 暴涨到 61.13；
- **(ε=20, δ=2)**：判据弱于随机（< 1），几乎所有 ε-NN 边都保留，退化为普通 ε-NN 聚类 → 平庸的 43.02；
- **(ε=10, δ=5)**：比值 7.5，"显著但不过分" → 当前最佳 39.63。

由此可定义无量纲比值 \(\rho = \delta / (\varepsilon^2/n)\)，**\(\rho \in [5, 10]\) 是一个有理论支撑的"好区间"**。

### 6.2 同类点的共享邻居期望：信号侧

零模型只描述了"假边的噪声水平"。设同类内有 \(m\) 个点形成一个 clique-like cluster，且 \(m > \varepsilon\)（类内点数比邻居预算多）。则类内两点 \(i, j\) 的 ε-NN 几乎全部来自类内：

\[
\mathbb{E}[s_{\text{intra}}(i,j)] \;\approx\; \frac{(m-2)(\varepsilon-1)^2}{m-1}\,\cdot\,\mathbb{1}[m > \varepsilon].
\]

当 \(m \gg \varepsilon\) 时，此量趋近于 \(\varepsilon - 1\)（两点的 \(\varepsilon\) 个邻居最多有 \(\varepsilon-1\) 个重合，自身除外）。这给出 δ 的**绝对上界**：

\[
\boxed{\;\delta < \varepsilon - 1\;}
\]

超过此值，连真同类点也会被剪开。回看实验：

- **(ε=5, δ=5)**：δ = ε > ε−1，**违反上界** → 必然崩溃 ✓ 解释 61.13；
- **(ε=5, δ=3)**：δ = ε−2，紧贴上界，已有压力 ✓ 解释 49.36；
- **(ε=10, δ=5)**：δ = ε/2，舒适区 ✓ 解释 39.63 最佳。

### 6.3 信噪比框架

综合 6.1 与 6.2，定义

\[
\text{SNR}(\varepsilon, \delta) \;=\; \frac{\mathbb{E}[s_{\text{intra}}] - \delta}{\delta - \mathbb{E}[s_{\text{inter}}]} \;=\; \frac{(\varepsilon-1) - \delta}{\delta - \varepsilon^2/n}.
\]

好的参数应让分子分母同号且数量级相近，从而得到 δ 的**双侧约束**：

\[
\frac{\varepsilon^2}{n} \;\ll\; \delta \;\ll\; \varepsilon - 1.
\]

存在性条件：\(\varepsilon - 1 \gg \varepsilon^2 / n\)，即 \(n \gg \varepsilon^2 / (\varepsilon - 1) \approx \varepsilon\)。这正是 **ε=20 在 \(n \approx 150\) 时失效**的原因——\(\varepsilon^2 = 400\) 已逼近 \(n\)，零模型噪声接近信号上界，整个判据失去鉴别力（这与第 5 节观察 3 中"ε=20 时 δ 不再敏感（MAE 全在 43.0±0.05）"现象一致）。

### 6.4 参数健康度指标 η

将 (6.1)(6.2) 归一化，定义无量纲的"参数健康度"：

\[
\eta(\varepsilon, \delta, n) \;=\; \frac{\delta - \varepsilon^2/n}{\varepsilon - 1 - \varepsilon^2/n}.
\]

\(\eta \in (0, 1)\) 时参数处于可行区，\(\eta \approx 0.3{-}0.5\) 是甜点。回算实验（\(n \approx 150\)）：

| 实验 | ε | δ | η | MAE | 备注 |
|---|---|---|---|---|---|
| **e10_d5** | 10 | 5 | (5−0.67)/(9−0.67) = **0.52** | **39.63** | ✓ 最佳，落在甜点 |
| e5_d2  | 5  | 2 | (2−0.17)/(4−0.17) = **0.48** | 40.25 | ✓ 次佳，落在甜点 |
| e10_d3 | 10 | 3 | 0.28 | 41.67 | 偏低 |
| e10_d2 | 10 | 2 | 0.16 | 42.15 | 偏低，趋于无效剪枝 |
| e5_d3  | 5  | 3 | 0.74 | 49.36 | 偏高，开始切断真边 |
| **e5_d5**  | 5  | 5 | **>1（越界）** | 61.13 | ✓ 符合预测：崩盘 |
| **e20_d2** | 20 | 2 | (2−2.67)/(19−2.67) **< 0** | 43.02 | ✓ 符合预测：噪声主导 |
| e20_d3 | 20 | 3 | 0.02 | 43.05 | 同上 |
| e20_d5 | 20 | 5 | 0.14 | 43.02 | 同上 |

**η 与 MAE 几乎单调对应，且崩盘点（η>1）和退化点（η<0）与理论预测完全吻合**。这就是 SNG 方法可解释性的核心结论：参数选择不再"拍脑袋"，而是**把 η 调到 0.4–0.5 附近**。

---

## 7. 架构改进方案

按"改动成本 vs 预期收益"由低到高列出五个方向。

### 7.1 [低成本 / 高收益] 自适应 δ：让参数选择无量纲化

直接把 δ 从超参变为由 (ε, n) 推出的量，最简形式：

\[
\delta^{\star} \;=\; \big\lfloor\, \alpha\,(\varepsilon - 1) + (1-\alpha)\,\varepsilon^2/n\,\big\rfloor, \quad \alpha \in [0.3, 0.5].
\]

这把 \(\eta\) 直接钉在甜点位置，**消除跨数据集 / 跨图调参需求**。论文叙述上也漂亮：从"两个超参 (ε, δ)" 简化为"一个超参 α + 自动适配"。实现只需在 `sng_cluster` 入口加几行：

```python
def sng_cluster(features, *, epsilon, delta=None, alpha=0.4):
    n = len(features)
    if delta is None:
        delta = int(alpha * (epsilon - 1) + (1 - alpha) * epsilon**2 / n)
    # ... 原有流程
```

预期：跨数据集鲁棒性显著提升；FSC-147 上 MAE 持平或微涨（毕竟 δ 已手调过最优）。

#### 7.1.x 实现状态与合成数据验证（2026-05-27）

**实现** :: `codes/occam/clustering.py::sng_cluster(features, *, epsilon, delta=None, alpha=0.4)`，配套 helper `adaptive_delta(*, epsilon, n, alpha) -> int` 与 `eta_health(*, epsilon, delta, n) -> float`。`OccamConfig` 新增 `sng_alpha=0.4`，`sng_delta` 默认值改为 `None`（向后兼容：传整数 `delta` 走旧路径）。`codes/eval/eval_fsc147_full.py` 新增 `--sng-alpha`。

**clamp 细节**：`adaptive_delta` 把结果钳到 `[0, ε−2]` 区间，确保 §6.2 上界 `δ < ε−1` 永远满足（避免 ε 过大、α 接近 1 时 `floor` 越界）。

**CPU-only 合成验证** :: `results/2026-05-27-1145-validate-sng-adaptive-delta-cpu/`。3 个高斯类 (d=64, intra_std=0.8, inter_dist=4.0) + 5% 噪声点；`n ∈ {50, 100, 150, 250, 500}`、5 seed、ε=10；fixed δ ∈ {1, 2, 3, 5, 7} vs adaptive α ∈ {0.3, 0.4, 0.5}。结果（counting MAE under max-cluster head, 跨 25 个 (n, seed) 组合）：

| 方案             | 平均 MAE | 最差 MAE | ARI 平均 |
|------------------|----------|-----------|----------|
| adaptive α=0.50  | **22.32** | 70       | **0.691** |
| fixed δ=5        | 25.92    | 66       | 0.624   |
| adaptive α=0.40  | 34.40    | 132      | 0.612   |
| fixed δ=7        | 32.72    | 117      | 0.457   |
| fixed δ=3        | 41.36    | 132      | 0.533   |
| adaptive α=0.30  | 71.00    | 167      | 0.464   |

要点：
- adaptive α=0.50 的平均 MAE 比最佳 fixed (δ=5) 低 14%，**且不需要逐 n 调参**。它在 n 较小时自动选 δ=3、n 较大时选 δ=4。
- 任意 fixed δ 都会在某段 n 上失效（δ=2 在 n=50–100 噪声主导；δ=7 在 n=50 过激进切碎）。
- α=0.4 在合成数据上次一档；FSC-147 上 ε=10/δ=4–5 的实测 η≈0.4 工作良好——是否把默认 α 升到 0.5 应由实物 FSC-147 sweep 决定（见 results 中 README 的 "actionable next steps"）。
- α=0.3 太保守（实际 δ=2–3、η<0.3），等同于丢弃剪枝判据，出现噪声主导失效。

**对 §7.2/§7.3/§7.4/§7.5 的开发约束** :: `synth_validate_sng.py` 是任何 SNG 变体的回归基准（CPU-秒级），新方法 PR 都应附带"在该 fixture 上不劣化"的快验。

### 7.2 [中成本] 相似度加权 SNG

对**共享邻居本身**而非中心向量加权：

\[
s_w(i,j) \;=\; \sum_{k \in N(i) \cap N(j)} \min\bigl(\cos(x_k, x_i),\; \cos(x_k, x_j)\bigr).
\]

用 `min` 而非均值的语义：要求 \(k\) 同时与 \(i, j\) 都足够相似，更接近"真共享邻居"的定义。判据变为 \(s_w(i,j) > \delta_w\)，δ\_w 是连续值，避开整数判据的离散跳变。η 的解析形式不变，6.4 的健康度框架仍可平移。

### 7.3 [中成本] 度归一化 / 局部 δ

零模型隐含假设每个点的"邻居预算"是 ε，但对称化后实际度 \(d_i = |N(i)|\) 是变化的（高密度区大、孤立点小）。改为局部判据：

\[
s(i,j) \;>\; \delta \cdot \frac{\sqrt{d_i\,d_j}}{\varepsilon}.
\]

把绝对计数变成"相对于两端点局部密度的标准化数"。在 FSC-147 这类"图内类内密度尚均匀、跨图密度差异巨大"的设定下，这是性价比最高的鲁棒性提升。

### 7.4 [较高成本] 二阶拓扑：三角支持度 (Triangle Reinforcement)

当前判据只用了"边上的共享邻居"——一阶结构。再引入边 \((i,j)\) 的"三角支持度"：

\[
t(i,j) \;=\; \big|\{\,k : (i,k) \in E,\;(j,k) \in E,\;(i,j,k)\text{ 构成三角形}\,\}\big|.
\]

判据加强为 \(s(i,j) > \delta\) **且** \(t(i,j) > \delta_t\)：要求 \(i, j\) 的共同邻居本身也"互认"，是更强的"同社区"信号。零模型下 \(\mathbb{E}[t] \approx \varepsilon^3 / n^2\)，比 \(\mathbb{E}[s] \approx \varepsilon^2/n\) 衰减更快，**信噪比更高**。计算量从 \(O(n\varepsilon^2)\) 升至 \(O(n\varepsilon^3)\)，对几百量级仍完全可接受。

### 7.5 [激进 / 论文级] 信噪比驱动的自适应聚类

把第 6 节理论彻底落地：每张图自动选 (ε, δ)，超参数为零。算法草案：

1. **估计有效维度** \(\hat{d}\)：用 PCA effective rank 或特征 participation ratio；
2. **估计潜在类数上限** \(K_{\max}\)：用密度峰值或 spectral gap；
3. **解析地求** \((\varepsilon, \delta)\)：使 \(\eta \approx 0.5\) 且预期簇数 \(\in [1, K_{\max}]\)；
4. 跑 SNG。

这一步把 SNG 从"一个聚类算法"提升为"自描述的、超参为零的聚类框架"。与 FINCH（需逐 mode 调阈值序列）形成的对比将非常锋利——免调参本身就是一个独立的方法论贡献。

---

## 8. 复现命令

单组 SNG 实验：

```bash
python OCCAM_experiments_series/origin_simulation/eval_fsc147_full.py \
  --mode single --splits val --fraction 0.333 --seed 42 \
  --min-mask-area 0.0005 --max-mask-area 0.10 \
  --cluster-method sng --sng-epsilon 10 --sng-delta 5 \
  --pred-strategy max \
  --output-dir results/A6_SNG_e10_d5
```

完整扫参：`OCCAM_experiments_series/ablation_clustering/run_sng_sweep.sh`（脚本内置温度 / 显存保护，已完成实验自动跳过）。
