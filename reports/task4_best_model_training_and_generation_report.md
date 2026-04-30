# Task 4 — Best Model Training and Sample Generation Report (Part 4)

## English

### Overview

This report covers **Part 4 (15%)**: we train the **best model** longer, generate SVG samples (unconditional + prefix-conditioned), and evaluate generation quality using quantitative metrics.

- **Chosen best model:** **SP (standard parameterization) XL** from Part 2.
- **Why SP XL:** It achieved the best validation loss among the models we trained in Parts 2–3, so we continued training it to improve likelihood and generation quality.

### Where the code and results live

- **Task 4 code:** `scripts/task4/`
  - `train_best_model.py`: longer training + checkpointing (`final.pt`, `summary.json`, `metrics.jsonl`)
  - `sample_generate.py`: unconditional + prefix-conditioned sampling; optional PNG rendering via CairoSVG
  - `evaluate_generation.py`: test perplexity + XML/structural validity + render rate
  - `figure_report.py`: rendered grid figure for the write-up
- **Colab notebooks (Task 4):**
  - `notebooks/task4_colab_best_model (1).ipynb`: first attempt (write outputs directly to Drive)
  - `notebooks/task4_colab_best_model_(2)_.ipynb`: final attempt (write locally first; uses **H100** runtime)
- **Outputs used in this report:**
  - Baseline run: `outputs/task4_1/`
  - Final run: `outputs/task4_final/`

### Two runs: baseline vs final (what changed and why)

We ended up with **two main Task 4 executions**:

1. **Baseline (`task4_1`)**: trained for **3 epochs** while writing checkpoints and artifacts under **Google Drive**.
2. **Final (`task4_final`)**: trained for **7 epochs** on a **Colab H100** runtime, writing to **local `/content`** first and only copying essential artifacts at the end (to avoid Drive quota / slow I/O).

#### Key motivation: Drive quota and throughput

When saving frequent large checkpoints directly to Drive, the folder can fill quickly (each XL checkpoint can be ~1 GB). This caused failures when later stages attempted to write additional outputs (e.g., evaluation JSON). The final workflow avoids this by:

- writing to **local disk** during training (`/content/...`)
- disabling intermediate checkpoint dumps
- copying only the final artifacts after completion

### Training setup (fixed across runs)

Both runs used the same model family and training protocol:

- **Model:** decoder-only Transformer LM, **SP XL**
- **Vocab:** `vocab_size = 3535` (effective vocab from the processed token stream)
- **Sequence length:** `block_size = 512`
- **Batching:** `tokens_per_batch = 32768` → `batch_size = 64`
- **Optimizer:** AdamW (`weight_decay = 0.1`)
- **Schedule:** cosine decay to `min_lr = 1e-5` with `warmup_steps = 2000`
- **Data:** Part 1/2 token stream, **~103M** train tokens (`train.jsonl`) and **~1.05M** val tokens (`val.jsonl`)

### Training hyperparameters and outcomes (baseline vs final)

All values below are taken from the saved `config.json` / `summary.json`.

| Run | Output dir | LR | Epochs | Steps | Val loss |
|---|---|---:|---:|---:|---:|
| **Baseline** (`task4_1`) | Drive-style output (see `outputs/task4_1/MANIFEST.txt`) | **0.01** | **3** | **9453** | **0.8734** |
| **Final** (`task4_final`) | local-first output (`/content/task4_outputs_e7/...`) | **0.003** | **7** | **22057** | **0.8231** |

**Improvement:** training longer with a smaller LR reduced validation loss from **0.8734 → 0.8231**.

### Generation setup (required samples)

We generated:

- **Unconditional samples:** **10** (starting from an `<svg` prefix)
- **Prefix-conditioned samples:** **5** (partial SVG prefixes; the model completes them)
- **Sampling controls:** temperatures **0.5, 0.8, 1.0**, plus **top-k = 50** and **top-p = 0.95**

Artifacts are written under:

- `outputs/task4_final/samples/svg/` (SVG text)
- `outputs/task4_final/samples/png/` (rendered PNGs when `--render` succeeds)
- `outputs/task4_final/samples/manifest.json` (per-sample metadata including `render_ok`)

### Quantitative evaluation (final run)

The final run includes a completed quantitative evaluation:

- `outputs/task4_final/eval_metrics.json`

Reported metrics:

- **Test perplexity:** **2.2869**
- **XML validity rate:** **0.60**
- **Structural validity rate:** **0.60**
- **Render rate:** **0.60**

Interpretation:

- A non-trivial fraction of generated outputs are valid SVG/XML and renderable.
- Prefix-conditioned generations are harder: because prefixes are intentionally **partial SVG fragments**, the model sometimes fails to close tags / emit a complete `<svg>...</svg>` document, which reduces XML validity and render success.

### Qualitative deliverable (rendered grid)

The report grid figure for visual inspection is saved as:

- `outputs/task4_final/figures_report/samples_grid.png` (and `.pdf`)

### Deliverables checklist (Part 4)

- **Best model chosen and trained longer:** SP XL (baseline 3 epochs → final 7 epochs)
- **≥10 unconditional samples:** in `outputs/task4_final/samples/svg/`
- **≥5 prefix-conditioned samples:** in `outputs/task4_final/samples/svg/`
- **Temperature + top-k/top-p sampling:** recorded in `manifest.json`
- **Quantitative metrics:** `outputs/task4_final/eval_metrics.json`
- **Rendered grid:** `outputs/task4_final/figures_report/samples_grid.png`

# Task 4 — Best Model Training and Sample Generation Report (Part 4)

## English

### Overview

This report covers **Part 4 (15%)**: we train the **best model** longer, generate SVG samples (unconditional + prefix-conditioned), and evaluate generation quality using quantitative metrics.

- **Chosen best model:** **SP (standard parameterization) XL** from Part 2.
- **Why SP XL:** It achieved the best validation loss among the models we trained in Parts 2–3, so we continued training it to improve likelihood and generation quality.

### Where the code and results live

- **Task 4 code:** `scripts/task4/`
  - `train_best_model.py`: longer training + checkpointing (`final.pt`, `summary.json`, `metrics.jsonl`)
  - `sample_generate.py`: unconditional + prefix-conditioned sampling; optional PNG rendering via CairoSVG
  - `evaluate_generation.py`: test perplexity + XML/structural validity + render rate
  - `figure_report.py`: rendered grid figure for the write-up
- **Colab notebooks (Task 4):**
  - `notebooks/task4_colab_best_model (1).ipynb`: first attempt (write outputs directly to Drive)
  - `notebooks/task4_colab_best_model_(2)_.ipynb`: final attempt (write locally first; uses **H100** runtime)
- **Outputs used in this report:**
  - Baseline run: `outputs/task4_1/`
  - Final run: `outputs/task4_final/`

### Two runs: baseline vs final (what changed and why)

We ended up with **two main Task 4 executions**:

1. **Baseline (`task4_1`)**: trained for **3 epochs** while writing checkpoints and artifacts under **Google Drive**.
2. **Final (`task4_final`)**: trained for **7 epochs** on a **Colab H100** runtime, writing to **local `/content`** first and only copying essential artifacts at the end (to avoid Drive quota / slow I/O).

#### Key motivation: Drive quota and throughput

When saving frequent large checkpoints directly to Drive, the folder can fill quickly (each XL checkpoint can be ~1 GB). This caused failures when later stages attempted to write additional outputs (e.g., evaluation JSON). The final workflow avoids this by:

- writing to **local disk** during training (`/content/...`)
- disabling intermediate checkpoint dumps
- copying only the final artifacts after completion

### Training setup (fixed across runs)

Both runs used the same model family and training protocol:

- **Model:** decoder-only Transformer LM, **SP XL**
- **Vocab:** `vocab_size = 3535` (effective vocab from the processed token stream)
- **Sequence length:** `block_size = 512`
- **Batching:** `tokens_per_batch = 32768` → `batch_size = 64`
- **Optimizer:** AdamW (`weight_decay = 0.1`)
- **Schedule:** cosine decay to `min_lr = 1e-5` with `warmup_steps = 2000`
- **Data:** Part 1/2 token stream, **~103M** train tokens (`train.jsonl`) and **~1.05M** val tokens (`val.jsonl`)

### Training hyperparameters and outcomes (baseline vs final)

All values below are taken from the saved `config.json` / `summary.json`.

| Run | Output dir | LR | Epochs | Steps | Val loss |
|---|---|---:|---:|---:|---:|
| **Baseline** (`task4_1`) | Drive-style output (see `outputs/task4_1/MANIFEST.txt`) | **0.01** | **3** | **9453** | **0.8734** |
| **Final** (`task4_final`) | local-first output (`/content/task4_outputs_e7/...`) | **0.003** | **7** | **22057** | **0.8231** |

**Improvement:** training longer with a smaller LR reduced validation loss from **0.8734 → 0.8231**.

### Generation setup (required samples)

We generated:

- **Unconditional samples:** **10** (starting from an `<svg` prefix)
- **Prefix-conditioned samples:** **5** (partial SVG prefixes; the model completes them)
- **Sampling controls:** temperatures **0.5, 0.8, 1.0**, plus **top-k = 50** and **top-p = 0.95**

Artifacts are written under:

- `outputs/task4_final/samples/svg/` (SVG text)
- `outputs/task4_final/samples/png/` (rendered PNGs when `--render` succeeds)
- `outputs/task4_final/samples/manifest.json` (per-sample metadata including `render_ok`)

### Quantitative evaluation (final run)

The final run includes a completed quantitative evaluation:

- `outputs/task4_final/eval_metrics.json`

Reported metrics:

- **Test perplexity:** **2.2869**
- **XML validity rate:** **0.60**
- **Structural validity rate:** **0.60**
- **Render rate:** **0.60**

Interpretation:

- A non-trivial fraction of generated outputs are valid SVG/XML and renderable.
- Prefix-conditioned generations are harder: because prefixes are intentionally **partial SVG fragments**, the model sometimes fails to close tags / emit a complete `<svg>...</svg>` document, which reduces XML validity and render success.

### Qualitative deliverable (rendered grid)

The report grid figure for visual inspection is saved as:

- `outputs/task4_final/figures_report/samples_grid.png` (and `.pdf`)

### Deliverables checklist (Part 4)

- **Best model chosen and trained longer:** SP XL (baseline 3 epochs → final 7 epochs)
- **≥10 unconditional samples:** in `outputs/task4_final/samples/svg/`
- **≥5 prefix-conditioned samples:** in `outputs/task4_final/samples/svg/`
- **Temperature + top-k/top-p sampling:** recorded in `manifest.json`
- **Quantitative metrics:** `outputs/task4_final/eval_metrics.json`
- **Rendered grid:** `outputs/task4_final/figures_report/samples_grid.png`

---
