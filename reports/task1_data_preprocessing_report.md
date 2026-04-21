# Task 1 — Data Preprocessing Report (Part 1)

## English

### Overview

This report summarizes **Task 1**: building a cleaned SVG corpus, training a BPE tokenizer, splitting data by file, and recording dataset statistics. Processing was implemented in `scripts/task1/preprocess_dataset.py` with cleaning in `svg_scaling/cleaning.py`. Artifacts are stored under `data/processed/`; Task 1 figures and example renders are under `outputs/task1/`.

### Cleaned corpus

- **What it is:** Each Hugging Face row is one SVG **file** (keyed by dataset id and `Filename`). Raw SVG strings are passed through `clean_svg()`: XML parse with comment stripping, removal of `<metadata>`, `<title>`, and `<desc>`, whitespace normalization, float rounding (1 decimal), and re-validation as well-formed XML. Sequences outside character length bounds are dropped before splitting.
- **Outputs:** `data/processed/train.jsonl`, `val.jsonl`, and `test.jsonl`. Each line is a JSON object with cleaned `svg`, `num_tokens` (after BPE), `source_dataset`, and `filename`.
- **Scale after cleaning:** **364,130** files retained after cleaning (`kept_files_after_clean` in `stats.json`).

### Tokenizer

- **Library:** **SentencePiece** BPE (`sentencepiece`), not Hugging Face `tokenizers`.
- **Training:** The model is trained **only on the training split** text (one SVG per line, newlines collapsed to spaces), then applied to all splits.
- **Vocabulary size:** **4096** (within the suggested **1k–8k** range).
- **Artifacts:** `data/processed/spm.model`, `data/processed/spm.vocab`.
- **Length filter:** Sequences with more than **2048** BPE tokens are dropped (`max_token_len`).

#### Vocabulary size rationale (4096)

We set the SentencePiece vocabulary to **4096** because it sits in the middle of the **1K–8K** course range and offers a practical trade-off for SVG text: large enough to represent frequent tags, attributes, and numeric fragments as compact subwords, yet small enough to keep sequence lengths and embedding memory reasonable for modest decoder-only LMs. Very small vocabs increase sequence length and rare-token noise; very large vocabs add parameters with diminishing returns for this preprocessing stage, so **4096** is a standard default choice and matches our pipeline configuration.

#### Optional steps not applied in this run

- **Canonical attribute ordering** was **not** implemented (the assignment marks it optional). Sorting attributes lexicographically on every element would further stabilize the text form but adds implementation complexity (namespaces, duplicate handling) with limited benefit for our BPE objective; we rely on `lxml` serialization instead.
- **Full Cairo rasterization filtering** was **not** used during preprocessing (`--render-check` off). We instead **validated CairoSVG on a random sample** of `train.jsonl` to confirm the toolchain. A full Cairo gate would drop any SVG Cairo cannot rasterize and would require **re-running the entire preprocessing pipeline** (cleaning counts, splits, BPE, `stats.json`, and figures). To run it on Windows, install GTK/Cairo, ensure Python can load `libcairo-2.dll` (see `README.md`), activate the same venv, then from the repo root run the same `preprocess_dataset.py` command as for the merged corpus but add **`--render-check`**. Expect **much longer** runtime; if train tokens fall below **~100M**, increase `--fonts-subsample` (or add data) and re-run until the target is met.

### Train / validation / test split

- **Ratios:** **98% / 1% / 1%** (train / val / test).
- **Granularity:** **By file** — the list of cleaned file records is shuffled with a fixed seed (**42**), then partitioned. There is **no** splitting inside a single SVG sequence.
- **Sources merged:** `starvector/svg-icons-simple`, `starvector/svg-emoji-simple`, and a **subsample** of `starvector/svg-fonts-simple` (train split, capped row count) so that total training tokens meet the course target (≥ ~100M BPE tokens).

### Dataset statistics (summary)

| Item | Value |
|------|--------|
| Vocabulary size | **4096** |
| Train tokens | **102,899,186** |
| Validation tokens | **1,049,271** |
| Test tokens | **1,042,877** |

**Sequence length distribution** is summarized for the **training set** after tokenization (BPE length = `num_tokens`):

- **Histogram:** Stored in `data/processed/stats.json` under `train_length_histogram` (`bin_edges`, `counts`, and `quantiles`).
- **Quantiles (train):** min **59**; 25% **154**; median **203**; 75% **306**; 90% **571**; 99% **1347**; max **2048** (sequences at the cap are allowed up to 2048 tokens before filtering).

**Sample counts before / after filtering:**

- **After cleaning (before train/val/test split):** **364,130** files in one pool.
- **After 98/1/1 split, before the 2048-token cap:** **356,847** train + **3,641** val + **3,642** test = **364,130** (each split is encoded with the trained BPE; counts here include rows that will still be dropped if `num_tokens` > 2048).
- **After BPE + ≤2048 token filter (`files_after_token_filter`):** **356,748** train | **3,638** val | **3,641** test.
- **Dropped by token cap (>2048 BPE tokens):** **99** train + **3** val + **1** test = **103** total.

Equivalently: **364,130 − 103 = 364,027** files remain in the three jsonl files combined.

### Figures and example visuals

- **Statistics figure:** Sequence-length distribution is plotted from `stats.json` as **`outputs/task1/train_length_histogram.png`** (and **`train_length_histogram.pdf`** if generated) for the report.
- **Example SVGs by complexity:** **`outputs/task1/gallery.html`** embeds inline SVGs at approximate quantiles **0.10 / 0.50 / 0.90** of `num_tokens` on the training set, with companion `.svg` files in the same folder. These illustrate short-, medium-, and long-BPE examples for documentation.

### Reproducibility

- Random seed for splitting: **42** (`stats.json` → `seed`).
- Full configuration is recorded in `data/processed/stats.json`.

---

## 中文翻译

### 概述

本报告总结**任务一**：构建清洗后的 SVG 语料、训练 BPE 分词器、按**文件**划分数据集，并整理数据统计。流程由 `scripts/task1/preprocess_dataset.py` 实现，清洗逻辑在 `svg_scaling/cleaning.py`。数据产物位于 `data/processed/`；任务一图表与示例渲染位于 `outputs/task1/`。

### 干净数据集（cleaned corpus）

- **含义：** Hugging Face 中每一行对应一个 SVG **文件**（由数据集名与 `Filename` 唯一标识）。原始字符串经 `clean_svg()`：XML 解析并去注释、去掉 `<metadata>` / `<title>` / `<desc>`、空白规范化、浮点数保留约一位小数、并再次校验为良构 XML。超出字符长度阈值的样本在划分前丢弃。
- **输出：** `data/processed/train.jsonl`、`val.jsonl`、`test.jsonl`。每行一条 JSON，含清洗后的 `svg`、BPE 后的 `num_tokens`、`source_dataset`、`filename` 等。
- **清洗后规模：** 共保留 **364,130** 个文件（见 `stats.json` 中 `kept_files_after_clean`）。

### 分词器（tokenizer）

- **库：** **SentencePiece** BPE（`sentencepiece`），未使用 Hugging Face `tokenizers`。
- **训练：** 仅在 **训练集**文本上训练（每个 SVG 一行），再用于各划分编码。
- **词表大小：** **4096**（落在建议的 **1k–8k** 范围内）。
- **产物：** `data/processed/spm.model`、`data/processed/spm.vocab`。
- **长度过滤：** BPE 长度超过 **2048** 的序列会被丢弃（`max_token_len`）。

#### 词表大小说明（4096）

我们将 SentencePiece 词表设为 **4096**，使其落在课程建议的 **1K–8K** 中段：对 SVG 文本而言，该规模能在「常见标签、属性与数字片段的子词表示」与「序列长度、嵌入显存」之间取得平衡；词表过小会拉长序列并放大罕见子词噪声，过大则增加参数而收益递减，**4096** 是常用默认并与本流水线配置一致。

#### 本流程未采用的可选项

- **规范化属性顺序（canonical attribute ordering）** 未实现（作业中为可选）。若对每个元素的属性做字典序排序可进一步固定文本形式，但实现上需处理命名空间与重复键等细节，收益相对有限；当前依赖 **`lxml` 序列化** 输出即可。
- **预处理阶段的全量 Cairo 栅格过滤** 未开启（未使用 `--render-check`）。我们改为对 **`train.jsonl` 随机抽样**做 CairoSVG 校验以确认环境可用。若要对**每一条**样本做 Cairo 过滤，需**重新跑完整预处理**（清洗统计、划分、BPE、`stats.json`、图表均需重算）。在 Windows 上需先装好 GTK/Cairo 并保证 Python 能加载 `libcairo-2.dll`（见 `README.md`），激活同一 venv，在仓库根目录使用与合并数据时**相同**的 `preprocess_dataset.py` 命令并追加 **`--render-check`**；耗时会**显著增加**；若 train token 低于约 **1 亿**，需提高 `--fonts-subsample`（或增数据）后重跑直至达标。

### Train / val / test 划分

- **比例：** **98% / 1% / 1%**。
- **粒度：** **按文件** — 清洗后的文件列表用固定随机种子（**42**）打乱后按比例切分，**不会**在单条 SVG 序列中间切开。
- **数据源：** 合并了 `starvector/svg-icons-simple`、`starvector/svg-emoji-simple`，并对 `starvector/svg-fonts-simple` 的 train 划分做了**子采样**（限制行数），以满足课程对训练 token 总量（约 ≥1 亿 BPE tokens）的要求。

### 数据统计（dataset statistics）

| 项目 | 数值 |
|------|------|
| 词表大小 | **4096** |
| Train token 数 | **102,899,186** |
| Val token 数 | **1,049,271** |
| Test token 数 | **1,042,877** |

**序列长度分布：** 在**训练集**上、分词之后统计（长度为 BPE token 数 `num_tokens`）：

- **直方图数据：** 保存在 `data/processed/stats.json` 的 `train_length_histogram`（含 `bin_edges`、`counts`、`quantiles`）。
- **分位数（训练集）：** 最小 **59**；25% **154**；中位数 **203**；75% **306**；90% **571**；99% **1347**；最大 **2048**（未超长过滤前允许至 2048）。

**过滤前后样本数：**

- **清洗后、划分前：** 共 **364,130** 个文件。
- **98/1/1 划分之后、2048 token 截断之前：** train **356,847** + val **3,641** + test **3,642** = **364,130**。
- **分词并完成 ≤2048 token 过滤之后（`files_after_token_filter`）：** train **356,748** | val **3,638** | test **3,641**。
- **因超过 BPE 长度上限丢弃：** train **99** + val **3** + test **1** = **103**。

合计保留 **364,130 − 103 = 364,027** 条，写入三个 jsonl。

### 数据统计图与样例图

- **统计图：** 由 `stats.json` 中的长度分布绘制的 **`outputs/task1/train_length_histogram.png`**（及若已生成的 **`train_length_histogram.pdf`**），用于报告中的序列长度分布展示。
- **样例图：** **`outputs/task1/gallery.html`** 内嵌三条不同 **BPE 长度分位数**（约 **0.10 / 0.50 / 0.90**）的训练样本 SVG，同目录下有对应 **`.svg`** 文件，用于展示不同复杂度示例。

### 可复现性

- 划分随机种子：**42**（见 `stats.json` 的 `seed`）。
- 完整统计与配置见 **`data/processed/stats.json`**。
