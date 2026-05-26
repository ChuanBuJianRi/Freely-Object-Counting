---
name: quantum-reading
trigger: |
  - User asks to read/analyze/summarize PDF, DOCX, TXT, MD files
  - File sizes > 50KB or line counts > 500
description: 量子阅读 Quantum Reading - AI Agent 智能速读技能，通过人眼式扫描策略平均节省 ~40% Token
category: document-processing
---

# 目标
1. **最小化 token** - 策略性阅读,非全文阅读
2. **保护主 context** - 在 subagent 运行
3. **支持深入** - 维护位置索引供按需查询

---

# 阅读策略

## Phase 0: 问题分析 (如用户提问)

提取关键词 → 直接定位相关内容 → 跳过无关部分

```bash
# 问题: "隐私保护措施有哪些?"
# 策略: grep "隐私|privacy|保护|措施" → 读匹配段落 ± 上下文
```

## Phase 1: 智能扫描 (总是先做)

### 1. 三段论
- 开头 50 行 (主旨/背景)
- 结尾 50-100 行 (总结/结论)  
- 中间采样 (>1000 行时,读中间 50 行)

### 2. 寻找总结标记
```bash
# 中文
grep -n "总结|概述|结论|因此|综上|关键|核心" file

# 英文  
grep -in "summary|conclusion|abstract|therefore|key point" file

# 结构
grep -n "^# \|^## \|^\* \|^[0-9]\+\." file  # 标题/列表
```

### 3. 格式利用
| 格式 | 优先级 | 查找方法 |
|------|--------|----------|
| 一级标题 `#` | P0 | `grep "^# "` |
| 粗体 `**text**` | P1 | `grep "\*\*.*\*\*"` |
| 列表项 `* -` | P1 | `grep "^[\*\-] "` |
| 引用 `>` | P2 | `grep "^> "` |

### 4. 关键词定位
```bash
grep -in "keyword" file           # 忽略大小写
grep -C 5 "keyword" file          # 含上下文
grep -E "word1|word2|word3" file  # 多词
```

### 5. 段落首尾句优先
- 学术写作: **首句 = 主题, 尾句 = 总结**
- 仅读每段首尾句,跳过中间 (除非问及细节)

### 6. 高价值内容
```bash
# 列表/数字
grep -n "^[0-9]\+[\.\)、]\|^第[一二三四五]\+[章节]" file

# 数据
grep -E "[0-9]+%|[0-9]+\.[0-9]+" file

# 因果/转折 (关键逻辑)
grep -iE "因为|导致|然而|但是|because|however|although" file

# 例证
grep -in "例如|比如|for example|such as" file
```

---

## Phase 2: 创建索引

```
📑 INDEX
═══════════════════════════════════

META: [行数] | [类型] | [语言]

SECTIONS (位置)
├─ Intro (lines 1-200)
├─ Method (lines 201-500)
└─ Conclusion (lines 900-1000)

TOPICS (内容)
├─ Privacy → lines [45, 230, 678]
├─ Security → lines [89, 450]

HIGH-VALUE ZONES
├─ Abstract: lines 10-50
├─ Summary: lines 950-1000
└─ Lists: lines [120, 340, 560]

HINTS
├─ Quick read: lines [10-50, 950-1000]
├─ For details on X: lines [Y-Z]
```

---

## Phase 3: 回答问题 (带着问题找答案)

### 问题分类 + 策略

| 类型 | 策略 | 工具 |
|------|------|------|
| 主旨 | 读首尾段 | `head -50` + `tail -50` |
| 细节 | 关键词定位 | `grep "keyword"` |
| 推断 | 读相关段,综合 | 索引 → 读多段 |
| 词义 | 读该句 ± 3 行 | `grep -C 3` |
| 态度 | 找形容词/评价词 | `grep -E "形容词pattern"` |

### 搜索层级
1. **精确**: `grep "exact phrase"`
2. **模糊**: `grep -i "keyword"`  
3. **扩展**: `grep -E "word1|synonym1|synonym2"`
4. **同义**: "benefit" → "advantage|strength|positive"

### 读取范围
- ✅ 先查索引 → 定位行号
- ✅ 仅读该段 ± 10 行上下文
- ✅ 不够再扩展到邻近段
- ❌ 不要一次读大段

---

# 文件类型处理

| 类型 | 策略 |
|------|------|
| **PDF** | `Read(pages="1-2")` + 尾页,>10 页时采样 |
| **DOCX** | `pandoc file.docx -t plain` 后按 TXT 处理 |
| **TXT/MD** | `wc -l` 检查长度,>1000 行时分段读 |

---

# 输出格式

```
📄 DOC ANALYSIS

File: [name] | Size: [pages/lines] | Type: [type]

Summary (2-3 句): 
[执行摘要]

Structure:
• Section 1 (loc)
• Section 2 (loc)

Topics: [主题列表]

✓ Index created. Ask about any section for details.

💰 Token saved: Read X lines of Y total (Z% reduction)
```

---

# 核心规则

- ❌ **禁止全文逐行读** (除非明确要求)
- ✅ **总是先扫描后深入**
- ✅ **保存索引供后续查询**
- ✅ **首次扫描 <500 行**
- ✅ **告知策略**: "我先扫描建索引..."
- ✅ **提及节省**: "读了 120/200 句 (40% 节省)"

---

# 速读检查清单

**读前**:
- [ ] 看标题/目录?
- [ ] 识别文档类型?
- [ ] 提取问题关键词?
- [ ] 规划阅读路径?

**读中**:
- [ ] 找总结词?
- [ ] 读段首尾?
- [ ] 跳过已知信息?
- [ ] 记录位置?

**读后**:
- [ ] 能 2-3 句概括?
- [ ] 知道细节在哪?
- [ ] 能回答问题?

---

# 避免陷阱

1. 线性阅读 → ✅ 跳跃阅读
2. 过度完美 → ✅ 抓大放小
3. 忽略结构 → ✅ 利用格式
4. 被动阅读 → ✅ 带问题读
5. 一次读完 → ✅ 分层读取

---

**记住**: 你在 SUBAGENT 中 - 保护主 agent 的 context 是首要任务。
