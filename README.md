# svg-scaling-laws-transformers

Decoder-only **Transformer language models** on **SVG** text (SentencePiece BPE), with **Part 1 data preprocessing**, **Part 2 scaling-law experiments** (LR sweep, 1-epoch runs, plots), and optional extensions (e.g. **μP** in `requirements.txt` for later parts).

## What’s in this repo

| Area | Contents |
|------|-----------|
| **Part 1 — data** | `scripts/task1/` (`preprocess_dataset.py`, `verify_dataset.py`, `render_svg_examples.py`, …). Artifacts: `data/processed/` (`train.jsonl`, `val.jsonl`, `test.jsonl`, `spm.model`, `stats.json`, …). |
| **Part 2 — scaling** | `scripts/task2/` (`train.py`, `lr_sweep.py`, `model.py`, `data.py`, `figure_report.py`, `plot_scaling.py`, `config_presets.py`). See **`scripts/task2/README.md`** for Colab-oriented commands. |
| **Part 3 — μP** | `scripts/task3/` (`train_mup.py`, `lr_sweep_mup.py`, `model_mup.py`, …). Notebook: **`notebooks/task3_colab_mup.ipynb`**. See **`scripts/task3/README.md`**. |
| **Outputs** | `outputs/task1/` (histogram, SVG gallery). `outputs/task2/<preset>/` (`config.json`, `summary.json`, `metrics.jsonl`), LR sweep dirs, **`outputs/task2/figures_report/`** (scaling / loss / throughput figures). |
| **Reports** | `reports/task1_data_preprocessing_report.md`, `reports/task2_transformer_scaling_report.md` |
| **Notebooks** | `notebooks/01_verify_dataset.ipynb`, `02_preprocess.ipynb`, `03_task2_colab_scaling.ipynb`, `task2_A100.ipynb`, `task2_T4.ipynb` |

**Course runs in this fork:** Part 1 pipeline produces **≥100M** BPE train tokens when merging the recommended Hugging Face splits (see Task 1 report). Part 2 was trained primarily on **Google Colab** (GPU); full **Tiny→XL** scaling at fixed **`tokens_per_batch=32768`** completed on **A100** after **T4 OOM** on the largest model (see Task 2 report).

---

## Environment (local)

1. **Python 3.10+** recommended (close to Colab).
2. Create a virtual environment:

```bash
python -m venv .venv
```

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

3. Install dependencies:

```bash
pip install -U pip
pip install -r requirements.txt
```

If **PyTorch** fails to install, follow [pytorch.org](https://pytorch.org) for your OS, then run `pip install -r requirements.txt` again. **GPU training** needs a CUDA build of PyTorch; CPU is fine only for smoke tests.

### Cairo on Windows (CairoSVG / `--render-check`)

`pip install cairosvg` only installs Python wrappers; **Cairo** must be available as **`libcairo-2.dll`**. Without it you may see `no library called "cairo-2" was found`.

1. **Install GTK3 Runtime (64-bit)** from [GTK-for-Windows Runtime installer releases](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases). Prefer **Win64** matching 64-bit Python; leave **“Set up PATH”** enabled.
2. **Restart** the terminal / IDE so `PATH` updates.
3. If needed, set the DLL folder explicitly (PowerShell; adjust path):

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES = 'C:\Program Files\GTK3-Runtime Win64\bin'
```

4. **Smoke-test** render validation on a sample:

```powershell
python scripts/task1/validate_render.py --jsonl data/processed/train.jsonl --max-samples 200
```

**Alternative (MSYS2):** `mingw-w64-x86_64-cairo` and add `C:\msys64\mingw64\bin` to `PATH` (or `CAIROCFFI_DLL_DIRECTORIES`).

---

## Part 1 — Verify access, preprocess, figures

### Verify dataset access

From the **repository root**:

```bash
python scripts/task1/verify_dataset.py
```

Or open `notebooks/01_verify_dataset.ipynb`.

### Clean, tokenize, split (core pipeline)

```bash
python scripts/task1/preprocess_dataset.py --output-dir data/processed
```

This merges configured Hugging Face sources, **cleans** SVGs, trains **SentencePiece** BPE (default vocab **4096**) on **train** only, drops sequences over **2048** BPE tokens, and writes `train.jsonl` / `val.jsonl` / `test.jsonl`, `spm.model`, `spm.vocab`, `stats.json`.

To match the course token budget, use the same dataset flags as in your report (e.g. icons + emoji + **subsampled fonts**); example:

```bash
python scripts/task1/preprocess_dataset.py --output-dir data/processed \
  --dataset starvector/svg-icons-simple \
  --extra-datasets starvector/svg-emoji-simple
```

Useful flags: `--vocab-size`, `--max-token-len`, `--seed`, `--render-check` (optional; slow). **Full narrative + statistics:** `reports/task1_data_preprocessing_report.md`.

### Task 1 figures (length histogram + SVG gallery)

```bash
python scripts/task1/render_svg_examples.py --jsonl data/processed/train.jsonl --out-dir outputs/task1
```

Writes **`gallery.html`** and example **`.svg`** files. **PNG** export works if **CairoSVG** (with Cairo DLL), **Inkscape**, or **ImageMagick** is available ([Inkscape](https://inkscape.org/release/) is often easiest on Windows).

**PowerShell:** long commands on one line, or use backtick `` ` `` for line continuation (not `^`).

---

## Part 2 — Transformer scaling (Colab / GPU)

Training code lives under **`scripts/task2/`**. Typical workflow:

1. **LR sweep on Tiny** (7 log-spaced learning rates, cosine + warmup per run):

```bash
python -m scripts.task2.lr_sweep \
  --train-jsonl data/processed/train.jsonl \
  --val-jsonl data/processed/val.jsonl \
  --spm-model data/processed/spm.model \
  --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
  --out-csv outputs/task2/lr_sweep.csv
```

2. **Train each preset** (1 epoch, fixed `tokens-per-batch` and `block-size`; only `--preset` changes):

```bash
python -m scripts.task2.train \
  --train-jsonl data/processed/train.jsonl \
  --val-jsonl data/processed/val.jsonl \
  --spm-model data/processed/spm.model \
  --preset tiny --lr <BEST_FROM_SWEEP> \
  --tokens-per-batch 32768 --block-size 512 \
  --out-dir outputs/task2/tiny
```

Repeat for `small`, `medium`, `large`, `xl`. **Large/XL** on **Colab T4 (~15GB)** may **OOM** at full batch tokens; **A100** (or smaller `--tokens-per-batch`, with report justification) is often required.

3. **Figures for the report** (scaling plot, train-loss grid, throughput / memory / wall-time bars, summary table):

```bash
python -m scripts.task2.figure_report \
  --task2-dir outputs/task2 \
  --out-dir outputs/task2/figures_report
```

**Colab:** use **`notebooks/task2_A100.ipynb`** / **`task2_T4.ipynb`** or **`03_task2_colab_scaling.ipynb`**; point `--train-jsonl`, `--val-jsonl`, `--spm-model` at your Drive or `/content` paths after uploading Part 1 artifacts.

**Write-up:** `reports/task2_transformer_scaling_report.md` (LR sweep, fair comparison table, power-law fit, metrics, compute notes).

**More detail:** `scripts/task2/README.md` (nanoGPT-style credit, metric fields in `metrics.jsonl`).

---

## Part 3 — μP scaling (`mup`)

Code: **`scripts/task3/`** (`model_mup.py`, `train_mup.py`, `lr_sweep_mup.py`, `plot_sp_vs_mup.py`, `config_presets.py`). **Read `scripts/task3/README.md`** for the Part 2 vs `MU_PRESETS` depth/heads note (required for honest SP vs μP plots).

**Colab notebook:** `notebooks/task3_colab_mup.ipynb` (clone → install `mup` → LR sweep → train all sizes → overlay SP vs μP → extrapolation sketch).

```bash
python -m scripts.task3.lr_sweep_mup --train-jsonl ... --val-jsonl ... --spm-model ... --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 --out-csv outputs/task3/lr_sweep_mup.csv
python -m scripts.task3.train_mup --preset xl --lr <BEST> --train-jsonl ... --out-dir outputs/task3/xl
python -m scripts.task3.plot_sp_vs_mup --task2-dir outputs/task2 --task3-dir outputs/task3 --out outputs/task3/figures/sp_vs_mup_scaling.png
```

---

## Reports (submission)

- **Part 1:** `reports/task1_data_preprocessing_report.md` — corpus, tokenizer, splits, statistics, figures.
- **Part 2:** `reports/task2_transformer_scaling_report.md` — LR sweep, scaling law, tables, Colab/T4/A100 notes.

---

## Google Colab (general)

1. **Runtime → Change runtime type → GPU** (T4 vs A100 affects max model / batch).
2. Clone this repo, `cd` into it, `pip install -r requirements.txt` (or install `torch`, `sentencepiece`, `numpy`, `matplotlib`, `scipy`, `tqdm` per `scripts/task2/README.md`).
3. Upload or mount **`data/processed/`** (large `jsonl` + `spm.model`); write **`outputs/`** under the clone or on Drive.

Large generated files stay under `data/`, `outputs/`, and are **gitignored** by default; keep a zip or Drive copy for reproducibility.
