# svg-scaling-laws-transformers

Scaling law study for decoder-only Transformer language models on SVG data, including preprocessing, LR sweeps, µP comparison, and sample generation.

## Environment (local)

1. **Python 3.10+** recommended (matches Colab closely).
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

If PyTorch fails to install, follow [https://pytorch.org](https://pytorch.org) for your OS, then run `pip install -r requirements.txt` again.

### Cairo on Windows (CairoSVG / `--render-check`)

`pip install cairosvg` only installs Python wrappers; **Cairo itself must be on your machine** (native `libcairo-2.dll`). Without it you will see errors like `no library called "cairo-2" was found`.

1. **Install GTK3 Runtime (64-bit)** from the [GTK-for-Windows Runtime installer releases](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases). Choose the **Win64** build that matches your Python (64-bit). During setup, **leave “Set up PATH” enabled** so `libcairo-2.dll` is discoverable.
2. **Close and reopen** Cursor / PowerShell / Terminal so the updated `PATH` is picked up.
3. If Cairo still is not found, point **cairocffi** at the DLL folder explicitly (PowerShell; adjust the path if your install location differs):

```powershell
$env:CAIROCFFI_DLL_DIRECTORIES = 'C:\Program Files\GTK3-Runtime Win64\bin'
```

4. **Check** render validation on a sample of your processed data:

```powershell
python scripts/validate_render.py --jsonl data/processed/train.jsonl --max-samples 200
```

**Alternative (MSYS2):** install `mingw-w64-x86_64-cairo` with `pacman` and add `C:\msys64\mingw64\bin` to PATH (or set `CAIROCFFI_DLL_DIRECTORIES` to that `bin` folder).

## Step 1 — Verify dataset access

From the **repository root**:

```bash
python scripts/verify_dataset.py
```

You should see split names, row counts, and a short preview of the `Svg` field from `starvector/svg-icons-simple`.

## Step 2 — Clean, tokenize, and split (Part 1 core)

From the **repository root**:

```bash
python scripts/preprocess_dataset.py --output-dir data/processed
```

This will:

- Merge unique files from `starvector/svg-icons-simple`, then shuffle and split **98% / 1% / 1%** by file ID.
- **Clean** SVGs (strip comments, remove `metadata`/`title`/`desc`, normalize whitespace, round decimals, re-validate XML).
- **Train** a SentencePiece BPE model (**4096** vocab by default) on the **train** split only.
- **Drop** sequences longer than **2048** tokens (per BPE), then write token counts per row.
- Save `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`, `spm.model`, `spm.vocab`, and `stats.json` (histogram + token totals).

The assignment asks for **≥100M training tokens**. With icons-only data you may see a **warning** and ~tens of millions of tokens. Merge extra Hugging Face datasets (same `Filename` / `Svg` schema), for example:

```bash
python scripts/preprocess_dataset.py --output-dir data/processed \
  --dataset starvector/svg-icons-simple \
  --extra-datasets starvector/svg-emoji-simple
```

Useful flags: `--vocab-size`, `--max-token-len`, `--seed`, `--render-check` (optional CairoSVG gate; slower).

### Report figures (rendered SVGs)

**What it means:** SVG source is plain text; “rendering” means turning it into a **picture** (PNG) so readers see the actual icon/shape. The course asks for examples at **different complexity levels** — use short vs long sequences (e.g. by `num_tokens` after BPE).

After `train.jsonl` exists:

```bash
python scripts/render_svg_examples.py --jsonl data/processed/train.jsonl --out-dir outputs/figures
```

This writes **SVG** files and **`gallery.html`** under `outputs/figures/`. **`gallery.html` embeds each SVG as base64**, so double‑clicking works in Edge/Chrome (older versions used `<object src=".svg">`, which `file://` often blocks).

**PNG files** are written if any of these works: **CairoSVG** (needs Cairo DLL), **Inkscape** on `PATH`, or **ImageMagick** (`magick` on `PATH`). On Windows, installing [Inkscape](https://inkscape.org/release/) is usually the easiest way to get PNG without Conda.

**PowerShell:** put the command on **one line**, or use a line break with backtick `` ` `` (not `^`):

```powershell
python scripts/preprocess_dataset.py --output-dir data/processed --dataset starvector/svg-icons-simple --extra-datasets starvector/svg-emoji-simple
```

## Google Colab

1. **Runtime → Change runtime type → GPU** (optional for this step).
2. Clone your fork/repo, `cd` into it, then either:
   - Open `notebooks/01_verify_dataset.ipynb` or `notebooks/02_preprocess.ipynb` and run all cells, or
   - Run `pip install -r requirements.txt` and the scripts under `scripts/` in code cells.

Large downloads and checkpoints should go under `data/`, `outputs/`, or `checkpoints/` (ignored by git by default).
