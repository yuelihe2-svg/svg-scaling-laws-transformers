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

