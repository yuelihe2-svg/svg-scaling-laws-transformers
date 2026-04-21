# Part 2 — Transformer scaling (decoder-only LM on SVG BPE)

This folder implements a **nanoGPT-style** causal Transformer (`scripts/task2/model.py`), data loading from Part-1 `jsonl` + SentencePiece (`scripts/task2/data.py`), and training (`scripts/task2/train.py`). It is meant to satisfy the course **scaling study**: LR sweep on **Tiny**, then one **1-epoch** run per size with **fixed** `tokens-per-batch`, cosine+warmup, AdamW.

Run all commands from the **repository root** so `python -m scripts.task2...` resolves.

## Do I need `.py` files if I use Colab?

**Yes.** Colab is just a remote machine: you normally **`git clone`** this repo and run the same Python modules. A **`.ipynb` notebook** is optional glue (install deps, set paths, call `!python -m scripts.task2.train ...`). All real logic should stay in **`scripts/task2/*.py`** so your report can cite reproducible code (and match the “nanoGPT-style + your changes” requirement).

## Google Colab workflow

1. **Runtime → Change runtime type → GPU** (T4 is fine for Tiny/Small/Medium; Large/XL may need smaller `--tokens-per-batch` or a larger GPU).

2. **Get the code**
   - `git clone <your-fork>` in Colab, or upload a zip of the repo.

3. **Get Part-1 data on Colab** (large files)
   - Zip `data/processed/train.jsonl`, `val.jsonl`, `test.jsonl`, `spm.model` (and `spm.vocab` if needed) from your PC, upload to **Google Drive**, unzip in Colab; **or** upload directly to `/content`.

4. **Install dependencies**
   ```bash
   pip install torch sentencepiece numpy matplotlib scipy tqdm
   ```
   (Use [pytorch.org](https://pytorch.org) Colab snippet if you want a specific CUDA wheel.)

5. **LR sweep (Tiny, 5–7 log-spaced LRs)**  
   From repo root:
   ```bash
   python -m scripts.task2.lr_sweep \
     --train-jsonl /content/data/processed/train.jsonl \
     --val-jsonl /content/data/processed/val.jsonl \
     --spm-model /content/data/processed/spm.model \
     --lrs 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 \
     --out-csv outputs/task2/lr_sweep.csv
   ```
   Pick the LR with lowest `val_loss` in the CSV. Use that **`--lr` for all presets**.

6. **Train each preset (1 epoch)**  
   Keep **`--tokens-per-batch`** and **`--block-size`** fixed across models; only change `--preset`:
   ```bash
   python -m scripts.task2.train \
     --train-jsonl ... --val-jsonl ... --spm-model ... \
     --preset small --lr <BEST_FROM_SWEEP> \
     --tokens-per-batch 32768 --block-size 512 \
     --out-dir outputs/task2/small
   ```
   Repeat for `medium`, `large`, `xl` (reduce `--tokens-per-batch` if OOM).

7. **Collect results**  
   Each run writes `outputs/.../summary.json` (`val_loss`, `n_params`). Build a JSON list, e.g. `points.json`:
   ```json
   [
     {"preset": "tiny", "n_params": 1712640, "val_loss": 3.12},
     {"preset": "small", "n_params": ..., "val_loss": ...}
   ]
   ```

8. **Scaling plot + power law**  
   ```bash
   python -m scripts.task2.plot_scaling --results points.json --out outputs/task2/scaling.png
   ```

9. **Metrics**  
   `metrics.jsonl` per run logs `train_loss`, `lr`, `tokens_per_s`, `gpu_mem_gb` (CUDA), wall time. Use these for the report tables (throughput, memory, curves).

## Notebook

See `notebooks/03_task2_colab_scaling.ipynb` for a minimal template (paths + example command).

## Credit / nanoGPT

This stack follows common **GPT-2 / nanoGPT** patterns (causal attention, AdamW, cosine schedule). Cite **nanoGPT** in your report and describe which files you adapted (`scripts/task2/model.py`, `scripts/task2/train.py`) versus wrote from scratch.

## Smoke test (CPU, tiny data)

```bash
python -m scripts.task2.train \
  --train-jsonl data/processed/train.jsonl --val-jsonl data/processed/val.jsonl \
  --spm-model data/processed/spm.model --preset tiny --max-docs 200 \
  --block-size 128 --tokens-per-batch 4096 --warmup-steps 10 \
  --out-dir outputs/task2/smoke --device cpu
```
