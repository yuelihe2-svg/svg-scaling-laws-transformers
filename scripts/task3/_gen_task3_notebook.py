"""One-off helper to write notebooks/task3_colab_mup.ipynb (run from repo root)."""

from __future__ import annotations

import json
from pathlib import Path


def md(s: str) -> dict:
    lines = s.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(s: str) -> dict:
    lines = s.strip().split("\n")
    src = [ln + "\n" for ln in lines]
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": src,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cells: list[dict] = []
    cells.append(
        md(
            """# Task 3 — μP scaling & SP comparison (Google Colab)

This notebook runs **Part 3** using the repository Python modules:

- `scripts/task3/lr_sweep_mup.py` — LR sweep on the **smallest μP** preset (default: `tiny`).
- `scripts/task3/train_mup.py` — train one width with `mup`, `MuAdamW`, and `set_base_shapes`.
- `scripts/task3/plot_sp_vs_mup.py` — overlay **Part 2 (SP)** vs **Part 3 (μP)** scaling curves.

## Before you start

1. **Runtime → Change runtime type → GPU** (A100 recommended for XL at full batch; T4 may OOM).
2. Put Part 1 artifacts on Drive (or `/content`): `train.jsonl`, `val.jsonl`, `spm.model`.
3. (Optional) Complete **Part 2** first so `outputs/task2/{tiny,...,xl}/summary.json` exist for the comparison plot.

## μP presets note

`MU_PRESETS` uses the **same depth (12 layers) and heads (8)** for every size so `mup.set_base_shapes` can match module trees across widths. **Part 2** used **different depths** per name; only **XL** parameter count matches Part 2 for the same vocabulary. **Explain this** when comparing curves in your report."""
        )
    )
    cells.append(
        md(
            """## 1) Clone repository and set `REPO_ROOT`

Set `TASK_REPO_URL` in the environment, or edit `CLONE_URL` below."""
        )
    )
    cells.append(
        code(
            r"""import os, subprocess, sys

CLONE_URL = os.environ.get("TASK_REPO_URL", "https://github.com/yuelihe2-svg/svg-scaling-laws-transformers.git")

os.chdir("/content")
print("cwd:", os.getcwd())
subprocess.run(["rm", "-rf", "svg-scaling-laws-transformers"], check=False)
subprocess.check_call(["git", "clone", CLONE_URL])

CANDIDATES = [
    "/content/svg-scaling-laws-transformers",
    "/content/svg-scaling-laws-transformers/svg-scaling-laws-transformers",
]
REPO_ROOT = None
for cand in CANDIDATES:
    marker = os.path.join(cand, "scripts", "task3", "train_mup.py")
    print(cand, "->", os.path.exists(marker))
    if REPO_ROOT is None and os.path.exists(marker):
        REPO_ROOT = cand

assert REPO_ROOT is not None, "Could not find scripts/task3/train_mup.py"
os.environ["REPO_ROOT"] = REPO_ROOT
os.chdir(REPO_ROOT)
print("REPO_ROOT:", REPO_ROOT)
print("cwd:", os.getcwd())"""
        )
    )
    cells.append(md("""## 2) Install dependencies (includes `mup`)"""))
    cells.append(
        code(
            r"""import subprocess, sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "sentencepiece", "numpy", "matplotlib", "scipy", "tqdm", "mup"]
)

import torch

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""
        )
    )
    cells.append(
        md(
            """## 3) Mount Google Drive and set `DATA_ROOT`

Change `DATA_ROOT` to the folder that contains `train.jsonl`, `val.jsonl`, and `spm.model`."""
        )
    )
    cells.append(
        code(
            r"""from google.colab import drive

drive.mount("/content/drive")

import os

DATA_ROOT = "/content/drive/MyDrive/svg_task2_data"
TRAIN = os.path.join(DATA_ROOT, "train.jsonl")
VAL = os.path.join(DATA_ROOT, "val.jsonl")
SPM = os.path.join(DATA_ROOT, "spm.model")

for p in (TRAIN, VAL, SPM):
    print(p, os.path.exists(p))

assert os.path.exists(TRAIN) and os.path.exists(VAL) and os.path.exists(SPM), "Missing data files under DATA_ROOT"
"""
        )
    )
    cells.append(
        md(
            """## 4) LR sweep on smallest μP model (7 rates)

Runtime is about **7× one Tiny epoch**. After this cell, `BEST_LR` is picked automatically from the CSV."""
        )
    )
    cells.append(
        code(
            r"""import csv, os, subprocess, sys

REPO_ROOT = os.environ["REPO_ROOT"]
OUT_CSV = os.path.join(REPO_ROOT, "outputs", "task3", "lr_sweep_mup.csv")

cmd = [
    sys.executable,
    "-m",
    "scripts.task3.lr_sweep_mup",
    "--train-jsonl",
    TRAIN,
    "--val-jsonl",
    VAL,
    "--spm-model",
    SPM,
    "--lrs",
    "1e-4",
    "3e-4",
    "1e-3",
    "3e-3",
    "1e-2",
    "3e-2",
    "1e-1",
    "--out-csv",
    OUT_CSV,
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=REPO_ROOT)

rows = list(csv.DictReader(open(OUT_CSV, encoding="utf-8")))
best = min(rows, key=lambda r: float(r["val_loss"]))
BEST_LR = float(best["lr"])
print("BEST_LR =", BEST_LR, "val_loss =", best["val_loss"])"""
        )
    )
    cells.append(
        md(
            """## 5) Train all μP presets (1 epoch, fixed tokens per batch)

Uses `BEST_LR` for every size (μTransfer protocol). **This is the longest step.**"""
        )
    )
    cells.append(
        code(
            r"""import os, subprocess, sys

REPO_ROOT = os.environ["REPO_ROOT"]
PRESETS = ["tiny", "small", "medium", "large", "xl"]

for preset in PRESETS:
    out_dir = os.path.join(REPO_ROOT, "outputs", "task3", preset)
    cmd = [
        sys.executable,
        "-m",
        "scripts.task3.train_mup",
        "--train-jsonl",
        TRAIN,
        "--val-jsonl",
        VAL,
        "--spm-model",
        SPM,
        "--preset",
        preset,
        "--lr",
        str(BEST_LR),
        "--tokens-per-batch",
        "32768",
        "--block-size",
        "512",
        "--warmup-steps",
        "2000",
        "--out-dir",
        out_dir,
    ]
    print("\n====", preset, "====")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO_ROOT)

print("All μP presets finished.")"""
        )
    )
    cells.append(
        md(
            """## 6) Plot Part 2 (SP) vs Part 3 (μP)

Requires both `outputs/task2/<preset>/summary.json` and `outputs/task3/<preset>/summary.json`."""
        )
    )
    cells.append(
        code(
            r"""import os, subprocess, sys

REPO_ROOT = os.environ["REPO_ROOT"]
out_png = os.path.join(REPO_ROOT, "outputs", "task3", "figures", "sp_vs_mup_scaling.png")
os.makedirs(os.path.dirname(out_png), exist_ok=True)

cmd = [
    sys.executable,
    "-m",
    "scripts.task3.plot_sp_vs_mup",
    "--task2-dir",
    os.path.join(REPO_ROOT, "outputs", "task2"),
    "--task3-dir",
    os.path.join(REPO_ROOT, "outputs", "task3"),
    "--out",
    out_png,
]
subprocess.check_call(cmd, cwd=REPO_ROOT)
print("Wrote", out_png)"""
        )
    )
    cells.append(md("""## 7) Extrapolation sketch (10× XL parameter count)

Simple power-law fit; the printed band is **not** a formal confidence interval — tighten this for your write-up."""))
    cells.append(
        code(
            r"""import json, os

import numpy as np
from scipy.optimize import least_squares

REPO_ROOT = os.environ["REPO_ROOT"]
presets = ["tiny", "small", "medium", "large", "xl"]


def load_curve(sub: str):
    root = os.path.join(REPO_ROOT, "outputs", sub)
    ns, ls = [], []
    for p in presets:
        path = os.path.join(root, p, "summary.json")
        s = json.load(open(path, encoding="utf-8"))
        ns.append(float(s["n_params"]))
        ls.append(float(s["val_loss"]))
    return np.array(ns, float), np.array(ls, float)


def fit_power(n, loss):
    x = np.log(n)
    y = loss

    def res(theta):
        a, alpha, c = theta
        return a * np.exp(-alpha * x) + c - y

    r = least_squares(
        res,
        [max(float(y.max() - y.min()), 1e-6), 0.3, float(y.min())],
        bounds=([0.0, 0.01, -np.inf], [np.inf, 3.0, np.inf]),
        max_nfev=2000,
    )
    return tuple(float(t) for t in r.x), float(np.sqrt(np.mean(r.fun**2)))


n2, l2 = load_curve("task2")
n3, l3 = load_curve("task3")
use_sp = float(l2.mean()) <= float(l3.mean())
n, l, tag = (n2, l2, "SP") if use_sp else (n3, l3, "muP")
(a, alpha, c), rmse = fit_power(n, l)
N_xl = float(n[-1])
N_pred = 10.0 * N_xl
L_pred = a * (N_pred ** (-alpha)) + c
print("Using", tag, "fit: a =", a, "alpha =", alpha, "c =", c, "rmse =", rmse)
print("XL N =", N_xl, "10x N =", N_pred)
print("Predicted val loss:", L_pred)
print("Naive RMSE band lower:", L_pred - rmse, "upper:", L_pred + rmse)"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }
    out = root / "notebooks" / "task3_colab_mup.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
