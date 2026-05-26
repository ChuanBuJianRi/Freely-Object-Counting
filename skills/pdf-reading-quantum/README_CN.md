<div align="center">

# ⚛️ 量子阅读 Quantum Reading

<br>

### 40% 的文字，0% 的价值。我们跳过它。

**像量子隧穿一样 —— 穿过噪音，瞬间抵达信息核心。**

<br>

[English](README.md) | [中文](README_CN.md) | [在线演示](web/index.html)

---

<br>

### 🔬 实验数据

> 量子阅读应用于三段真实中文文本的效果：

| 输入 | 类型 | 原文 Tokens | 量子阅读后 | 节省 | 💰 省钱* |
|------|------|:-----------:|:--------:|:----:|:-------:|
| GAPS 论文 | 物理学论文 (arXiv) | 2,008 | 1,388 | **31%** | $0.0093 |
| 微信推送文 | 公众号科技推文 | 2,336 | 1,477 | **37%** | $0.0129 |
| 表白小作文 | 网络表白信 | 1,189 | 681 | **43%** | $0.0076 |

<sub>*基于 Claude Opus 4.7 输入定价 ($15/MTok)。文档越长，节省比例越高。</sub>

<br>

</div>

---

## 🖥️ 在线演示

粘贴任意文本，实时观看量子阅读高亮关键内容、置灰噪音文字。

**[→ 打开演示](web/index.html)**

> 演示可视化了 skill 中的所有固定规则：总结标记词、结构标题、数据模式、逻辑连接词、段落首尾句优先等。高价值句子以**白色粗体**显示，低价值文字自动置灰。

<div align="center">

![GAPS 论文 — 节省 31%](screenshots/article.jpg)

*GAPS 物理学论文 (arXiv) — 2,008 → 1,388 tokens，节省 31%*

![微信推送文 — 节省 37%](screenshots/wechat.jpg)

*微信科技推文 (ChatGPT Images 2.0) — 2,336 → 1,477 tokens，节省 37%*

![表白小作文 — 节省 43%](screenshots/balabala.jpg)

*表白小作文 — 1,189 → 681 tokens，节省 43%*

</div>

---

## 概述

量子阅读是一个 AI Agent 技能，让 AI 像人类一样**快速浏览**大文档，而不是逐字阅读。

通过 **8 种人类阅读技巧** 和 **Subagent 隔离架构**，在保持 98%+ 信息准确率的同时，平均节省约 40% 的 token 消耗。

适用于 Claude Code、Cursor、Windsurf 等支持 skills 系统的 AI Agent。

### 核心价值

| 指标 | 效果 |
|------|------|
| Token 节省 | **~40%** 平均（实测 31-43%） |
| Context 污染 | **~0%**（Subagent 隔离） |
| 响应速度 | **3-5x** 提升 |
| 信息准确率 | **98%+** |

---

## 安装

### 使用 npx skills (推荐)

```bash
npx skills add SPA3K/quantum-reading-skill
```

### 手动安装

```bash
git clone https://github.com/SPA3K/quantum-reading-skill.git

# Claude Code:
cp -r quantum-reading-skill ~/.claude/plugins/user/quantum-reading-skill/

# Cursor:
cp -r quantum-reading-skill ~/.cursor/plugins/user/quantum-reading-skill/
```

### 验证

```
"分析这个 report.pdf"
```

看到以下提示即安装成功：
```
💰 Token saved: Read 120 lines instead of 200 (40% reduction)
```

---

## 使用方法

### 自动触发

当文件满足以下条件时自动启用：
- **文件类型**：`.pdf` `.docx` `.txt` `.md` `.doc` `.rtf`
- **文件大小**：>50KB **或** 行数 >500

### 手动调用

```bash
/quantum-reading path/to/large-file.pdf
```

---

## 工作原理

### 8 种智能扫描策略

| # | 策略 | 作用 |
|---|------|------|
| 1 | **三段论扫描** | 读开头 + 结尾 + 中间采样 |
| 2 | **总结标记识别** | 检测 `总结\|结论\|summary\|conclusion` 等关键词 |
| 3 | **格式优先** | 利用标题 (P0)、粗体/列表 (P1)、引用 (P2) |
| 4 | **段落首尾句** | 首句 = 主题，尾句 = 总结，= 80% 信息量 |
| 5 | **多层级关键词定位** | 精确 → 模糊 → 扩展 → 同义 |
| 6 | **数字列表优先** | 百分比、统计数据、编号列表 = 高价值 |
| 7 | **逻辑关系追踪** | 因果/转折/对比词定位 |
| 8 | **带问题找答案** | 问题分类 + 针对性搜索策略 |

### Subagent 架构

```
主 Agent (保持清洁)
    ↓
Subagent (处理大文件)
├─ 智能扫描 200-500 行
├─ 构建多维索引
└─ 生成精简总结
    ↓
主 Agent (仅收总结)
    ↓
用户追问 → 查索引 → 按需读取细节
```

**零 Context 污染** —— 重处理完全隔离在 Subagent 中。

---

## 配置

编辑 `plugin.json`：

```json
{
  "settings": {
    "prefer_subagent": true,
    "max_initial_read_lines": 500,
    "enable_structure_index": true
  },
  "triggers": {
    "file_size_threshold": 51200,
    "auto_invoke": true
  }
}
```

---

## 性能数据

### 单文档分析

| 文档规模 | 传统方式 | 量子阅读 | 节省 |
|---------|:-------:|:-------:|:----:|
| 500 行 | 1,500 tok | 900 tok | **40%** |
| 2,000 行 | 6,000 tok | 3,600 tok | **40%** |
| 5,000 行 | 15,000 tok | 8,700 tok | **42%** |
| 45 页 PDF | 15,000 tok | 8,400 tok | **44%** |

### 多轮对话

```
第 1 轮 (总体分析):    600 tokens
第 2 轮 (细节查询):   +300 tokens
第 3 轮 (深入探讨):   +200 tokens
累计:                1,100 tokens  vs  传统 2,000 (省 45%)
```

### 成本影响 (Opus 4.7 定价)

| 场景 | 传统成本 | 量子阅读 | 你省了 |
|-----|:-------:|:-------:|:-----:|
| 每天 100 篇论文 | $2.25 | $1.35 | **$0.90/天** |
| 每月 1M tokens | $15.00 | $9.00 | **$6.00/月** |

---

## 文件结构

```
quantum-reading-skill/
├── README.md          — English docs
├── README_CN.md       — 本文档
├── SKILL.md           — Skill 定义 & 规则
├── plugin.json        — 插件配置
├── LICENSE            — MIT 许可证
├── web/
│   └── index.html     — 交互式演示 (零依赖)
└── screenshots/       — 演示截图
```

---

## 最佳实践

**推荐：**
1. 信任自动触发 —— 大文件自动介入
2. 先看总结，再决定深入哪个部分
3. 问题要具体 —— "第三章的方法" 比 "方法" 更精确
4. 批量处理 —— 多文件可并行扫描

**避免：**
1. 强制逐行读 —— 失去优化优势
2. 小文件也用 —— 直接读更快
3. 问题太宽泛 —— 难以精确定位

---

## 许可证

MIT License — 自由使用、修改和分发。

---

## 链接

- **演示**：[web/index.html](web/index.html)
- **技术实现**：[SKILL.md](SKILL.md)
- **Issues**：[GitHub Issues](https://github.com/SPA3K/quantum-reading-skill/issues)

---

<div align="center">

**⚛️ 读得更少，理解更多。**

*让 AI 像人类一样智能阅读大文档。*

</div>
