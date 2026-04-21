# Task 2 — Transformer Scaling Study Report (Part 2)

## English

### Overview

This report summarizes **Task 2 (35%)**: training a **family of decoder-only transformer language models** on **SVG token sequences** from Part 1, measuring **validation loss after exactly one epoch** per size, performing a **learning-rate sweep on the smallest model**, keeping training setup **fair across sizes**, fitting a **power law** in the number of parameters, and reporting **throughput, memory, wall time, and training curves**.

- **Implementation:** `scripts/task2/` (`model.py`, `data.py`, `train.py`, `lr_sweep.py`, `config_presets.py`, `figure_report.py`, `plot_scaling.py`).
- **Primary experiment outputs:** `outputs/task2/<preset>/` (`config.json`, `summary.json`, `metrics.jsonl`).
- **LR sweep aggregate:** `outputs/task2/lr_sweep.csv` (plus per-LR runs under `outputs/task2/sweep_tiny_lr*/`).
- **Report figures (generated for submission):** `outputs/task2/figures_report/` (`scaling.png`, `train_loss_all.png`, `bars_time_throughput_memory.png`, `summary_table.png`, PDFs, `points.json`, `scaling_fit.txt`).
- **Reproduction notebooks (Google Colab):** `notebooks/task2_A100.ipynb` (successful full runs), `notebooks/task2_T4.ipynb` (initial attempts; see **Challenges**).

### Compute environment and challenges

- **Local machine:** No suitable **NVIDIA GPU** was available for multi-hour training at the required **tokens-per-batch** and **XL** width; all serious training was done on **Google Colab** with GPU runtime.
- **Initial Colab GPU (T4, ~15 GB VRAM):** Runs were **slow** relative to A100-class hardware. On the **largest preset (XL)**, training failed with **`torch.OutOfMemoryError: CUDA out of memory`**: the process had allocated on the order of **~14.5 GiB**, leaving only tens of MiB free when PyTorch tried to allocate an additional **~96 MiB** (typical Colab T4 headroom exhaustion under large attention/activations at fixed batch token count). This matches the course guidance that **Large/XL may need a larger GPU or reduced `--tokens-per-batch`**; we chose **not** to shrink batch tokens for the scaling comparison (assignment asks for **constant tokens-per-batch**), and instead **switched to an A100** Colab runtime to complete **XL** and finalize metrics.
- **Final Colab GPU (A100):** All five presets completed **1 epoch** with **`tokens_per_batch = 32768`**, **`block_size = 512`**, and full logging to `metrics.jsonl`.

### Code credit (nanoGPT / course requirement)

The stack follows common **GPT-2 / nanoGPT-style** patterns: **causal self-attention**, **AdamW** (`betas=(0.9, 0.95)`), **cosine learning rate** with **linear warmup**, and a compact decoder-only LM layout. **nanoGPT** is the conceptual reference; our **SVG-specific** pieces are **data loading** from Part-1 `jsonl` + **SentencePiece** (`scripts/task2/data.py`), **preset definitions** (`config_presets.py`), and **experiment drivers** (`train.py`, `lr_sweep.py`, reporting scripts). See also `scripts/task2/README.md`.

### Model sizes (at least five)

We trained **five** presets aligned with the course table (`scripts/task2/config_presets.py`). Actual parameter counts include embeddings for the **data-derived vocabulary** (see `config.json` → `n_params`, `vocab_size`).

| Preset | ~Params (spec) | d_model | n_layers | n_heads | d_ff | Actual n_params | Vocab (model) |
|--------|------------------|---------|----------|---------|------|-----------------|---------------|
| Tiny   | ~1M  | 128 | 4  | 4  | 512  | **1,761,792** | 3535 |
| Small  | ~3M  | 192 | 6  | 6  | 768  | **4,120,704** | 3535 |
| Medium | ~10M | 384 | 6  | 6  | 1536 | **13,549,824** | 3535 |
| Large  | ~30M | 512 | 10 | 8  | 2048 | **35,386,368** | 3535 |
| XL     | ~88M | 768 | 12 | 12 | 3072 | **90,842,112** | 3535 |

*Note:* Part 1 trained SentencePiece with **vocab_size 4096**, but the **integer token IDs present in the jsonl stream** yield a smaller **effective vocabulary** in the model (`vocab_size` above), which slightly shifts absolute parameter counts versus a naive “theory” table.

### Requirement 1 — Learning rate sweep (Tiny)

- **Model:** **Tiny** only (`n_params = 1,761,792`).
- **Schedule:** **Cosine decay with linear warmup** (same schedule family as full training; `warmup_steps = 2000` in full runs).
- **Learning rates:** **7** values, **approximately log-spaced**: `1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1` (recorded in `lr_sweep.csv` and under `outputs/task2/sweep_tiny_lr*/`).
- **Selection criterion:** **Lowest validation loss** in the sweep table.

| lr | val_loss (Tiny sweep) |
|----|------------------------|
| 0.0001 | 2.2656 |
| 0.0003 | 1.8808 |
| 0.001  | 1.5128 |
| 0.003  | 1.3221 |
| **0.01** | **1.1849** (best) |
| 0.03   | 1.2173 |
| 0.1    | 1.3177 |

**Chosen learning rate:** **`lr = 0.01`** for **all** scaling runs (Tiny → XL).

*Note:* The standalone **1-epoch Tiny** run’s `summary.json` reports `val_loss ≈ 1.202`, slightly above the sweep row at `0.01`; this is consistent with **different stochastic runs** (data order, dropout if any, numerics) while still using the **same selected LR**.

### Requirement 2 — Consistent training setup (fair comparison)

The following were held **constant across all five presets** (verified in each `outputs/task2/<preset>/config.json`):

| Setting | Value |
|---------|--------|
| Tokenization | Same **SentencePiece** model as Part 1 (`spm.model` on Colab / `data/processed/spm.model` locally) |
| Optimizer | **AdamW** (`weight_decay = 0.1`, `betas = (0.9, 0.95)`) |
| LR schedule | **Cosine** to **`min_lr = 1e-5`** with **`warmup_steps = 2000`** |
| Base LR | **`0.01`** (from Tiny sweep) |
| Sequence length | **`block_size = 512`** |
| Batch (tokens) | **`tokens_per_batch = 32768`** → `batch_size = 64` |
| Training duration | **`epochs = 1`**, **`max_steps = 3151`** (one pass over the capped stream configuration) |
| Train tokens (logged) | **`103,255,934` ≥ 100M** |
| Val tokens (logged) | **`1,052,909`** |
| Seed | **42** (default in `train.py`) |

### Requirement 3 — Scaling plot and power law

- **X-axis:** Parameter count **N** on a **log scale**.
- **Y-axis:** **Validation loss after 1 epoch** (from each `summary.json`).
- **Fit:** \( L = a N^{-\alpha} + c \) using nonlinear least squares (see `scripts/task2/figure_report.py` / `scaling_fit.txt`).

**Fitted values** (`outputs/task2/figures_report/scaling_fit.txt`):

- \(a \approx 59.80\)
- **\(\alpha \approx 0.376\)**
- \(c \approx 0.932\)
- RMSE \(\approx 0.0106\) on the five points

**Interpretation (brief):** Over the observed range (**~1.8M → ~9.1e7** parameters), validation loss **decreases with model size**; the fitted **\(\alpha\)** describes how steeply loss falls as \(N\) grows in this **power-law-plus-floor** parameterization, and **\(c\)** acts like a **residual floor** (irreducible error from data/noise/under-training at 1 epoch). With only **five** \((N, L)\) points, the fit is **illustrative**; **extrapolation** far outside this \(N\) range is **not reliable**.

**Figure:** `outputs/task2/figures_report/scaling.png` (and `scaling.pdf`).

### Requirement 4 — Additional metrics

1. **Training loss vs time/step:** `outputs/task2/figures_report/train_loss_all.png` (subplots per preset; source: each `metrics.jsonl` → `train_loss`, `step`).
2. **Wall-clock time per epoch:** recorded in `summary.json` → `wall_time_epoch_s`; summarized in `outputs/task2/figures_report/bars_time_throughput_memory.png` (left panel).
3. **GPU memory:** `metrics.jsonl` logs **`gpu_mem_gb`** each step; we report **peak** over the run in the table below and in the **right** panel of the bars figure.
4. **Throughput (tokens/s):** logged each step as `tokens_per_s`; table uses the **mean after step 200** (warmup/stabilization) to avoid inflating early partial steps.

### Results table (1 epoch, A100-class run)

| Preset | n_params | val_loss | wall_time_epoch_s | mean tok/s (step≥200) | peak GPU GB |
|--------|----------|----------|-------------------|------------------------|-------------|
| Tiny   | 1,761,792   | 1.2021 | 104.1  | 3,399,658 | 2.80 |
| Small  | 4,120,704   | 1.1138 | 228.1  | 2,341,062 | 4.09 |
| Medium | 13,549,824  | 1.0725 | 557.5  | 2,289,004 | 6.50 |
| Large  | 35,386,368  | 1.0072 | 1417.3 | 1,510,433 | 12.28 |
| XL     | 90,842,112  | 0.9942 | 3410.5 | 1,331,640 | 20.97 |

### Analysis — how loss changes with size

Validation loss **monotonically decreases** from **Tiny (1.20)** to **XL (0.99)** under identical data and token budget (1 epoch). Gains are **largest from Tiny→Small/Medium**, then **diminishing** toward XL—consistent with a **scaling-law** picture where extra capacity helps until optimization / one-epoch budget / residual error dominates. The fitted **\(\alpha\)** quantifies the slope on log–linear axes in the chosen functional form; comparing to published scaling studies should be done cautiously because **task (SVG LM), tokenization, and training budget** differ from standard web-scale LMs.

### Deliverables checklist (course)

1. **LR sweep (smallest model):** `outputs/task2/lr_sweep.csv` + `sweep_tiny_lr*/`.
2. **Scaling plot + fit:** `outputs/task2/figures_report/scaling.png`, `scaling_fit.txt`.
3. **Training curves (all models):** `outputs/task2/figures_report/train_loss_all.png`.
4. **Architecture + training statistics table:** this document + `outputs/task2/figures_report/summary_table.png`.
5. **Written analysis:** sections **Interpretation** and **Analysis** above.

### Reproducibility

- **Repo root commands** are documented in `scripts/task2/README.md`.
- **Regenerate figures:**  
  `python -m scripts.task2.figure_report --task2-dir outputs/task2 --out-dir outputs/task2/figures_report`

---

## 中文翻译

### 概述

本报告对应 **任务二（Part 2，约 35%）**：在 **Part 1 的 SVG BPE 序列**上训练 **多种规模的仅解码器 Transformer 语言模型**，对每个规模记录 **恰好 1 个 epoch 后的验证集 loss**；在 **最小模型（Tiny）** 上做 **学习率扫描**；各尺寸保持 **可比训练配置**；对参数量做 **幂律拟合** \(L=aN^{-\alpha}+c\) 并讨论 **\(\alpha\)**；并汇总 **训练曲线、墙钟时间、显存、吞吐（tokens/s）** 等指标。

- **代码：** `scripts/task2/`（`model.py`、`data.py`、`train.py`、`lr_sweep.py`、`config_presets.py`、`figure_report.py`、`plot_scaling.py`）。
- **主实验输出：** `outputs/task2/<preset>/`（`config.json`、`summary.json`、`metrics.jsonl`）。
- **学习率扫描汇总：** `outputs/task2/lr_sweep.csv`（以及各 `sweep_tiny_lr*/` 目录）。
- **报告用图：** `outputs/task2/figures_report/`（`scaling.png`、`train_loss_all.png`、`bars_time_throughput_memory.png`、`summary_table.png` 及 PDF、`points.json`、`scaling_fit.txt`）。
- **Colab 记录：** `notebooks/task2_A100.ipynb`（A100 上完整跑通）、`notebooks/task2_T4.ipynb`（T4 上尝试与瓶颈，见下文）。

### 计算环境与遇到的困难

- **本地环境：** 本机 **无可用 NVIDIA GPU**（或不足以支撑作业要求的批量 token 与 XL 规模下的长时间训练），因此主要实验在 **Google Colab GPU** 上完成。
- **初期使用 Colab T4（约 15GB 显存）：** 训练 **速度较慢**；在训练 **最大规模 XL** 时出现 **`torch.OutOfMemoryError: CUDA out of memory`**：进程已占用约 **14.5 GiB** 级别显存，仅剩约 **数十 MiB** 空闲时，PyTorch 再申请约 **96 MiB** 失败——这与 **大模型 + 固定 `tokens_per_batch`** 下的激活/注意力显存压力一致。课程说明也提示 **Large/XL 可能需要更大 GPU或减小 `--tokens-per-batch`**。为遵守作业 **「各尺寸 batch 以 tokens 计量保持一致」** 的要求，我们 **未** 通过减小每步 token 数来“凑”T4，而是 **更换为 A100** Colab 运行时，从而 **在相同 `tokens_per_batch=32768`、`block_size=512` 下完成 XL**，并保证与其他尺寸 **公平可比**。
- **最终 A100：** 五个 preset 均在上述固定配置下完成 **1 epoch**，指标写入 `metrics.jsonl` 与 `summary.json`。

### 代码来源说明（nanoGPT）

实现遵循常见 **GPT-2 / nanoGPT** 结构：**因果注意力**、**AdamW**、**warmup + cosine** 学习率等。**nanoGPT** 为方法与工程上的参考；与 **SVG 数据**强相关的部分主要为 **`jsonl` + SentencePiece 数据管道**（`data.py`）、**预设结构**（`config_presets.py`）以及 **训练/扫描脚本**（`train.py`、`lr_sweep.py` 等）。详见 `scripts/task2/README.md`。

### 模型规模（至少五种）

共 **5** 种规模，与课程建议表一致；**实际参数量**见各 `config.json` 的 `n_params`（含词嵌入，词表大小由数据中出现 token 决定，见上表 `vocab_size`）。

### 要求 1 — 最小模型上的学习率扫描（Tiny）

- **扫描对象：** 仅 **Tiny**。
- **学习率个数与间隔：** **7 个**，近似 **对数均匀**：`1e-4 … 1e-1`（见 `lr_sweep.csv`）。
- **调度：** 与正式训练一致族：**cosine + warmup**（正式跑 `warmup_steps=2000`）。
- **选择标准：** 以 **验证 loss 最低** 为准 → 选中 **`lr = 0.01`**。
- **一致性：** **Small→XL** 全部使用该学习率。

（扫描表见英文部分表格；注意：独立完整 1 epoch 的 Tiny `summary.json` 验证 loss 与扫描表中 `0.01` 一行可有小幅差异，属 **不同随机运行** 的正常现象。）

### 要求 2 — 统一训练设置（公平对比）

各 `config.json` 中一致的主要项：**同一 SentencePiece**、**AdamW**、**cosine+warmup**、**`lr=0.01`**、**`block_size=512`**、**`tokens_per_batch=32768`**、**`epochs=1`**、**`max_steps=3151`**、训练 token **约 1.03×10^8（≥1 亿）**、验证 token 约 **1.05×10^6**、随机种子 **42**。详见英文部分汇总表。

### 要求 3 — 缩放图与幂律

- **横轴：** 参数量 **N（对数坐标）**。
- **纵轴：** **1 epoch 后验证 loss**。
- **拟合：** \(L=aN^{-\alpha}+c\)，系数见 `scaling_fit.txt`：**\(\alpha \approx 0.376\)** 等。
- **讨论：** 规模增大验证 loss **总体下降**；**\(\alpha\)** 描述该区间内 loss 随 \(N\) 下降的“陡峭程度”；**\(c\)** 可理解为 **残差下界/不可约部分** 的建模。仅有 **5 个 (N,L) 点**，拟合用于 **展示趋势**；对 **未训练过的更大 N** 做 **外推预测** 需谨慎。

**图文件：** `outputs/task2/figures_report/scaling.png`。

### 要求 4 — 额外指标

1. **训练 loss 曲线：** `outputs/task2/figures_report/train_loss_all.png`（来源：各 `metrics.jsonl`）。
2. **每 epoch 墙钟时间：** `summary.json` 的 `wall_time_epoch_s`；见柱状图左栏。
3. **GPU 显存：** `metrics.jsonl` 的 `gpu_mem_gb`，表中为 **峰值**；见柱状图右栏。
4. **吞吐（tokens/s）：** `metrics.jsonl` 的 `tokens_per_s`；表中为 **step≥200 后的均值**（减少启动阶段波动）。

**汇总柱状图：** `outputs/task2/figures_report/bars_time_throughput_memory.png`。  
**一表汇总图：** `outputs/task2/figures_report/summary_table.png`。

### 结果表（1 epoch，A100 跑通）

（数值与英文部分 **Results table** 相同。）

### 分析 — loss 如何随规模变化

在 **相同数据与 1 epoch 预算** 下，验证 loss 从 Tiny 到 XL **单调下降**，但 **边际收益递减**（尤其接近 XL），与 **幂律 + 残差项** 的直观图像一致。与文献中通用 LM scaling 对比时，应强调 **任务（SVG）、分词与训练步数** 不同，**不宜直接套用**大模型公开曲线。

### 作业交付对照

1. 最小模型 LR 扫描结果：`lr_sweep.csv` 与 `sweep_tiny_lr*/`。  
2. 缩放图 + 拟合：`figures_report/scaling.png`、`scaling_fit.txt`。  
3. 各模型训练曲线：`figures_report/train_loss_all.png`。  
4. 架构与训练统计表：本报告表格 + `summary_table.png`。  
5. 文字分析：见 **Interpretation / Analysis** 与中文 **讨论** 小节。

### 可复现性

- 命令见 `scripts/task2/README.md`。  
- 重画图：`python -m scripts.task2.figure_report --task2-dir outputs/task2 --out-dir outputs/task2/figures_report`。
