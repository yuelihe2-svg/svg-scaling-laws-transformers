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


