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

## Step 1 — Verify dataset access

From the **repository root**:

```bash
python scripts/verify_dataset.py
```

You should see split names, row counts, and a short preview of the `Svg` field from `starvector/svg-icons-simple`.

## Google Colab

1. **Runtime → Change runtime type → GPU** (optional for this step).
2. Clone your fork/repo, `cd` into it, then either:
   - Open `notebooks/01_verify_dataset.ipynb` and run all cells, or
   - Run `pip install -r requirements.txt` and `python scripts/verify_dataset.py` in a code cell.

Large downloads and checkpoints should go under `data/`, `outputs/`, or `checkpoints/` (ignored by git by default).
