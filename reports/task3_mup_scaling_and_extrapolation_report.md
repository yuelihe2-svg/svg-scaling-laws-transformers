# Task 3 — μP Scaling and Extrapolation Report (Part 3)

## English

### Overview

This report covers **Part 3 (μP Scaling and Extrapolation)**. Starting from the Part 2 decoder-only Transformer LM on SVG BPE tokens, we:

- **Reparameterize** the model using **μP** (Maximal Update Parameterization) with the [`mup`](https://github.com/microsoft/mup) package.
- Perform a **learning-rate sweep** on the **smallest μP model** (same sweep protocol as Part 2).
- **Transfer** the best μP learning rate to larger μP models **without retuning**.
- Train **all model sizes** for **exactly 1 epoch** with the **same data, tokenization, and tokens-per-batch** as Part 2.
- **Compare SP (Part 2)** vs **μP (Part 3)** scaling curves and fit power laws to both.
- Use the best-fitting scaling law to **extrapolate** validation loss for a model with **10×** the parameters of the largest trained model.

Key reference: Yang et al. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero‑Shot Hyperparameter Transfer*.

### Where the code and results live

- **μP code:** `scripts/task3/`
  - `model_mup.py`: μP transformer (MuReadout, 1/d attention, base-shape setup)
  - `train_mup.py`: training loop using `MuAdamW` with cosine+warmup, 1 epoch
  - `lr_sweep_mup.py`: LR sweep runner
  - `plot_sp_vs_mup.py`: overlay plot of SP vs μP scaling curves
  - `figure_report.py`: one-shot figure generator for this report
- **Part 2 (SP) results:** `outputs/task2/`
- **Part 3 (μP) results:** `outputs/task3/`
- **Figures for this report:** `outputs/task3/figures_report/`

### μP implementation details (what changed vs Part 2)

The assignment notes two key transformer-specific changes for μP:

- **Attention scaling:** use **\(1/d\)** instead of **\(1/\sqrt{d}\)** (here \(d\) is the head dimension).
- **μP reparameterization:** use the `mup` package to apply width-dependent scaling so that hyperparameters (notably LR) can transfer across widths.

In our implementation:

- The LM output head is **`MuReadout(..., readout_zero_init=True)`**.
- We attach μP shape metadata by calling **`mup.set_base_shapes(model, base, delta)`** and train with **`mup.optim.MuAdamW`**.
- The LR schedule is applied **multiplicatively** to each optimizer group so μP’s relative LR scaling is preserved (important caveat from the `mup` README).

### Important preset note (fairness / interpretation)

`mup.set_base_shapes` requires the model module tree to match across widths. To satisfy this, the μP ladder (`scripts/task3/config_presets.py`) uses a **fixed depth and heads** across sizes:

- **μP presets:** `n_layers = 12`, `n_heads = 8` for every size; we scale mainly **width** (`d_model`, `d_ff`).
- **Part 2 presets (SP):** depths vary by size (Tiny 4, Small 6, Medium 6, Large 10, XL 12).

Therefore, **SP vs μP** curves are **not** “the exact same architecture with only parametrization changed”; the comparison is still valid for the course goal (hyperparameter transfer & scaling behavior), but the report must **explicitly state this**.

### 1) Learning-rate sweep on the smallest μP model

We ran a 1‑epoch LR sweep on the smallest μP model (named `tiny` in Part 3, but deeper than Part 2 Tiny). Results are saved to:

- `outputs/task3/lr_sweep_mup.csv`

The tested learning rates (log-spaced) match Part 2:

\[
10^{-4},\ 3\cdot10^{-4},\ 10^{-3},\ 3\cdot10^{-3},\ 10^{-2},\ 3\cdot10^{-2},\ 10^{-1}
\]

**Best μP learning rate (lowest val loss):** **`lr = 0.03`**.

We also plot **SP sweep (Part 2 Tiny)** vs **μP sweep (Part 3 Tiny μP)**:

- `outputs/task3/figures_report/lr_sweep_sp_vs_mup.png`

### 2) Train all μP sizes for exactly 1 epoch (no LR retuning)

For each μP preset (`tiny`, `small`, `medium`, `large`, `xl`), we trained for **1 epoch** using:

- Same Part 1 tokenization (`spm.model`)
- Same training data stream (≈ **103M** tokens)
- Same `block_size = 512`
- Same `tokens_per_batch = 32768` (batch size measured in tokens)
- Optimizer: **AdamW** via **`MuAdamW`**
- LR schedule: **cosine with warmup** (`warmup_steps = 2000`)

Each run writes:

- `outputs/task3/<preset>/config.json`
- `outputs/task3/<preset>/summary.json`
- `outputs/task3/<preset>/metrics.jsonl` (per-step logs; usually large)

### 3) Scaling comparison: SP (Part 2) vs μP (Part 3)

We overlaid both scaling curves and fit the required power law:

\[
L(N) = a\cdot N^{-\alpha} + c
\]

**Figure:** `outputs/task3/figures_report/sp_vs_mup_scaling.png`  
**Fit coefficients:** `outputs/task3/figures_report/sp_fit.txt`, `mup_fit.txt`

Fitted exponents:

- **SP (Part 2):** \(\alpha \approx 0.376\)
- **μP (Part 3):** \(\alpha \approx 0.238\)

Empirically, on this run:

- μP achieves **lower loss on the smallest scale** (its “tiny” starts lower than SP Tiny), but
- μP is **worse on the largest scales** (e.g., XL val loss is higher than SP XL).

Given the **architecture mismatch** (μP ladder uses 12 layers everywhere, SP ladder uses fewer layers for smaller names), the most honest interpretation is:

- This Part 3 run demonstrates the **required μP pipeline** (reparameterize → sweep → transfer LR → train all sizes → plot and fit).
- Whether μP “improves scaling” here is **inconclusive** as a pure parametrization comparison, because the ladders differ in depth/head configuration; additionally, μP attention uses a non-fused attention implementation (for explicit \(1/d\) scaling), which can affect optimization dynamics and throughput.

### 4) Scaling-law extrapolation (10× XL parameters)

Per the assignment, we extrapolate from the better of the two fits (here chosen by lower RMSE on the 5 points):

- Details: `outputs/task3/figures_report/extrapolation_10x_xl.txt`

Using the **better-RMSE curve** (μP in this run), for:

- \(N_{\text{XL}} = 90{,}842{,}112\)
- \(N_{\text{pred}} = 10\times N_{\text{XL}} = 908{,}421{,}120\)

we predict:

- **Predicted val loss:** \(\approx 0.998\)
- A rough **95% interval** is included in the text file (delta-method from the fitted Jacobian).

**Confidence discussion:** With only **5 points** and a narrow architecture/data regime (1 epoch only), uncertainty is substantial. Extrapolation can deviate due to:

- Changes in optimization at larger width/depth (e.g., different stability regions)
- Finite data effects / one-epoch undertraining (compute budget not scaled with model size)
- Implementation differences (attention kernel, μP base/delta choice)
- Tokenization/vocab and sequence statistics limiting the achievable floor \(c\)

### Deliverables checklist (Part 3)

- **μP LR sweep results:** `outputs/task3/lr_sweep_mup.csv` (+ figure `lr_sweep_sp_vs_mup.png`)
- **SP vs μP comparison plot:** `outputs/task3/figures_report/sp_vs_mup_scaling.png`
- **Power-law fits:** `outputs/task3/figures_report/sp_fit.txt`, `mup_fit.txt`
- **Extrapolation + uncertainty discussion:** `outputs/task3/figures_report/extrapolation_10x_xl.txt` + section above

---

## 中文

### 概述

本报告对应 **Part 3（μP 缩放与外推）**。在 Part 2 的 SVG BPE decoder-only Transformer 基础上，我们完成：

- 使用 `mup` 将模型改为 **μP（Maximal Update Parameterization）**；
- 在最小 μP 模型上做 **学习率扫描**（协议同 Part 2）；
- 将最优学习率 **直接迁移**到更大 μP 模型（不重新调参）；
- 在相同数据/分词/每步 token 数下，各尺寸训练 **恰好 1 个 epoch**；
- 将 **SP（Part 2）** 与 **μP（Part 3）** 的 scaling 曲线画在同一张图上，并分别拟合 \(L=aN^{-\alpha}+c\)；
- 用拟合较好的 scaling law 对 **10× XL 参数量**做验证 loss 外推，并讨论不确定性。

### 代码与输出位置

- μP 代码：`scripts/task3/`（`model_mup.py`, `train_mup.py`, `lr_sweep_mup.py`, `plot_sp_vs_mup.py`, `figure_report.py`）  
- Part 2 输出：`outputs/task2/`  
- Part 3 输出：`outputs/task3/`  
- 本报告图：`outputs/task3/figures_report/`

### μP 实现要点（相对 Part 2 的变化）

作业强调 Transformer 的 μP 实现需包含：

- 注意力缩放从 **\(1/\sqrt{d}\)** 改为 **\(1/d\)**（\(d\) 为 head 维度）；  
- 用 `mup` 完成 μP 重参数化，使得学习率可跨宽度迁移。

我们的实现中：

- LM head 使用 `MuReadout(..., readout_zero_init=True)`；
- 通过 `mup.set_base_shapes(model, base, delta)` 注入 μP 形状信息，并用 `MuAdamW` 优化；
- 余弦学习率调度以“乘法缩放”的方式作用到每个 param group，保持 μP 的相对 LR 缩放关系（`mup` 文档的关键注意事项）。

### 重要说明：preset 结构与 Part 2 不完全一致

为了满足 `set_base_shapes` 的“模块树一致”要求，Part 3 的 μP ladder 采用：

- **所有尺寸固定 `n_layers=12`、`n_heads=8`**，主要扩大 `d_model`/`d_ff`；

而 Part 2 的 SP ladder 深度随尺寸变化（4/6/6/10/12）。因此：

- **SP vs μP 并不是严格“同一架构仅换参数化”**，写报告时必须说明这一点。

### 1）最小 μP 模型学习率扫描

扫描结果写入 `outputs/task3/lr_sweep_mup.csv`，学习率同 Part 2 的 7 个对数点。最优学习率为：

- **`lr = 0.03`**（验证 loss 最低）

并输出对比图（SP Tiny sweep vs μP Tiny sweep）：

- `outputs/task3/figures_report/lr_sweep_sp_vs_mup.png`

### 2）用最优 LR 迁移训练所有 μP 尺寸（各 1 epoch）

所有 μP 尺寸使用相同数据/分词/每步 token 数，并训练 1 epoch，输出位于：

- `outputs/task3/<preset>/summary.json` 与 `config.json`

### 3）SP vs μP scaling 曲线与幂律拟合

缩放对比图：

- `outputs/task3/figures_report/sp_vs_mup_scaling.png`

拟合结果（\(L=aN^{-\alpha}+c\)）：

- SP：`sp_fit.txt`，\(\alpha \approx 0.376\)
- μP：`mup_fit.txt`，\(\alpha \approx 0.238\)

本次结果表现为：μP 在小模型起点更低，但在大模型端（如 XL）不如 SP。考虑到两条 ladder 的深度/head 不同，这一现象不能简单归因为“参数化优劣”，需要在报告中谨慎解释。

### 4）10× XL 外推与不确定性

外推文件：

- `outputs/task3/figures_report/extrapolation_10x_xl.txt`

本次按 RMSE 选择拟合更好的曲线（μP）进行外推，得到 10×XL 参数量下的 val loss 预测及一个粗略区间。由于只有 5 个点、并且仅训练 1 epoch，外推不确定性很大；偏差来源包括优化稳定性变化、有限数据/训练预算、实现细节差异等。

### 交付物对照

- μP 学习率扫描：`outputs/task3/lr_sweep_mup.csv` + `lr_sweep_sp_vs_mup.png`  
- SP vs μP 对比图：`sp_vs_mup_scaling.png`  
- 两条曲线幂律拟合：`sp_fit.txt`、`mup_fit.txt`  
- 10×XL 外推与不确定性讨论：`extrapolation_10x_xl.txt` + 本报告讨论段

