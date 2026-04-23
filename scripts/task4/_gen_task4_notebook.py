"""Generate notebooks/task4_colab_best_model.ipynb (run from repo root)."""

from __future__ import annotations

import json
from pathlib import Path


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in s.strip().split("\n")]}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [ln + "\n" for ln in s.strip().split("\n")],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cells: list[dict] = []

    cells.append(
        md(
            """# Task 4 — Best model training + sampling + evaluation (Colab)

This notebook trains the **chosen best model** longer (default: **SP XL from Part 2 settings**), then:

- Generates **≥10 unconditional** samples (from an `<svg` prefix)
- Generates **≥5 prefix-conditioned** completions
- Uses **temperature** + **top-k/top-p** sampling
- Computes quantitative metrics: **test perplexity**, **XML validity**, **structural validity**, **render rate**
- Builds a **rendered grid** figure for the report

By default, Task 4 artifacts (checkpoints, samples, eval JSON, report figures) are written under
`/content/drive/MyDrive/svg_task4_outputs/...` so a runtime disconnect is less likely to delete them.
The repo is still `REPO_ROOT` (cloned in section 1) and is used to run the Python modules.
"""
        )
    )

    cells.append(md("## 1) Clone repo & set `REPO_ROOT`"))
    cells.append(
        code(
            r"""import os, subprocess, sys

CLONE_URL = os.environ.get("TASK_REPO_URL", "https://github.com/yuelihe2-svg/svg-scaling-laws-transformers.git")

os.chdir("/content")
subprocess.run(["rm", "-rf", "svg-scaling-laws-transformers"], check=False)
subprocess.check_call(["git", "clone", CLONE_URL])

REPO_ROOT = "/content/svg-scaling-laws-transformers"
assert os.path.exists(os.path.join(REPO_ROOT, "scripts", "task4", "train_best_model.py"))
os.environ["REPO_ROOT"] = REPO_ROOT
os.chdir(REPO_ROOT)
print("REPO_ROOT:", REPO_ROOT)
print("cwd:", os.getcwd())"""
        )
    )

    cells.append(md("## 2) Install deps"))
    cells.append(
        code(
            r"""import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))"""
        )
    )

    cells.append(md("## 3) Mount Drive and set Part 1 data paths"))
    cells.append(
        code(
            r"""from google.colab import drive
drive.mount("/content/drive")

import os
DATA_ROOT = "/content/drive/MyDrive/svg_task2_data"  # change if needed
TRAIN = os.path.join(DATA_ROOT, "train.jsonl")
VAL = os.path.join(DATA_ROOT, "val.jsonl")
TEST = os.path.join(DATA_ROOT, "test.jsonl")  # if you saved it; otherwise use data/processed/test.jsonl from repo
SPM = os.path.join(DATA_ROOT, "spm.model")

# Part 4 artifacts (keep consistent across training / sampling / eval cells)
TASK4_OUT = "/content/drive/MyDrive/svg_task4_outputs"
MODEL_DIR = os.path.join(TASK4_OUT, "best_xl_sp")
SAMPLES_DIR = os.path.join(TASK4_OUT, "samples")
FIG_DIR = os.path.join(TASK4_OUT, "figures_report")
EVAL_JSON = os.path.join(TASK4_OUT, "eval_metrics.json")

for p in (TRAIN, VAL, SPM):
    print(p, os.path.exists(p))
assert os.path.exists(TRAIN) and os.path.exists(VAL) and os.path.exists(SPM)"""
        )
    )

    cells.append(md("## 4) Train best model longer (SP XL) + checkpoints"))
    cells.append(
        code(
            r"""import os, sys, subprocess
from glob import glob
REPO_ROOT = os.environ["REPO_ROOT"]

# Must match `MODEL_DIR` from the Drive cell above
OUT_DIR = MODEL_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Auto-resume: prefer the latest step_XXXXXX.pt; fallback to final.pt if present.
ckpt_dir = os.path.join(OUT_DIR, "checkpoints")
resume_ckpt = None
if os.path.isdir(ckpt_dir):
    step_ckpts = sorted(glob(os.path.join(ckpt_dir, "step_*.pt")))
    if step_ckpts:
        resume_ckpt = step_ckpts[-1]
    elif os.path.exists(os.path.join(ckpt_dir, "final.pt")):
        resume_ckpt = os.path.join(ckpt_dir, "final.pt")

cmd = [
    sys.executable, "-u", "-m", "scripts.task4.train_best_model",
    "--train-jsonl", TRAIN,
    "--val-jsonl", VAL,
    "--spm-model", SPM,
    "--preset", "xl",
    "--lr", "0.01",
    "--tokens-per-batch", "32768",
    "--block-size", "512",
    "--epochs", "3",
    "--save-every-steps", "500",
    "--out-dir", OUT_DIR,
]
if resume_ckpt is not None:
    print("Resuming from", resume_ckpt)
    cmd.extend(["--resume", resume_ckpt])
else:
    print("Starting fresh training")
print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=REPO_ROOT)
print("Done. OUT_DIR=", OUT_DIR)"""
        )
    )

    cells.append(md("## 5) Generate samples (SVG + PNG)"))
    cells.append(
        code(
            r"""import os, sys, subprocess
REPO_ROOT = os.environ["REPO_ROOT"]

os.makedirs(SAMPLES_DIR, exist_ok=True)

CKPT = os.path.join(MODEL_DIR, "checkpoints", "final.pt")
CFG = os.path.join(MODEL_DIR, "config.json")

cmd = [
    sys.executable, "-u", "-m", "scripts.task4.sample_generate",
    "--spm-model", SPM,
    "--checkpoint", CKPT,
    "--config", CFG,
    "--out-dir", SAMPLES_DIR,
    "--num-uncond", "10",
    "--num-prefix", "5",
    "--temperatures", "0.5", "0.8", "1.0",
    "--top-k", "50",
    "--top-p", "0.95",
    "--render",
]
print("Using MODEL_DIR =", MODEL_DIR)
print("Running:", " ".join(cmd))
subprocess.check_call(cmd, cwd=REPO_ROOT)"""
        )
    )

    cells.append(md("## 6) Quantitative evaluation (test perplexity + validity rates)"))
    cells.append(
        code(
            r"""import os, sys, subprocess
REPO_ROOT = os.environ["REPO_ROOT"]

CKPT = os.path.join(MODEL_DIR, "checkpoints", "final.pt")
CFG = os.path.join(MODEL_DIR, "config.json")

TEST_PATH = TEST if os.path.exists(TEST) else os.path.join(REPO_ROOT, "data", "processed", "test.jsonl")
print("Using TEST_PATH =", TEST_PATH)
print("Using MODEL_DIR =", MODEL_DIR)

cmd = [
    sys.executable, "-u", "-m", "scripts.task4.evaluate_generation",
    "--test-jsonl", TEST_PATH,
    "--spm-model", SPM,
    "--checkpoint", CKPT,
    "--config", CFG,
    "--samples-dir", SAMPLES_DIR,
    "--out-json", EVAL_JSON,
]
subprocess.check_call(cmd, cwd=REPO_ROOT)"""
        )
    )

    cells.append(md("## 7) Rendered grid figure for the report"))
    cells.append(
        code(
            r"""import os, sys, subprocess
REPO_ROOT = os.environ["REPO_ROOT"]
os.makedirs(FIG_DIR, exist_ok=True)

cmd = [sys.executable, "-u", "-m", "scripts.task4.figure_report", "--samples-dir", SAMPLES_DIR, "--out-dir", FIG_DIR]
subprocess.check_call(cmd, cwd=REPO_ROOT)"""
        )
    )

    cells.append(md("## 8) (Optional) Back up a copy into `.../task4_repo_mirror`"))
    cells.append(
        code(
            r"""# This notebook already writes Task 4 artifacts to Drive by default.
# Use this if you *also* want a second copy of `REPO_ROOT/outputs/task4` (local disk) for convenience.
!mkdir -p "/content/drive/MyDrive/svg_task4_outputs"
!rm -rf "/content/drive/MyDrive/svg_task4_outputs/task4_repo_mirror"
!cp -r "$REPO_ROOT/outputs/task4" "/content/drive/MyDrive/svg_task4_outputs/task4_repo_mirror"
print("Optional mirror: /content/drive/MyDrive/svg_task4_outputs/task4_repo_mirror")"""
        )
    )

    cells.append(md("## 9) Bundle the necessary artifacts into a single ZIP"))
    cells.append(
        md(
            """This creates a compact ZIP in `/content/` that you can download.
If Drive has space, it can also copy the ZIP into `TASK4_OUT/`."""
        )
    )
    cells.append(
        code(
            r"""import os, shutil
from pathlib import Path
from datetime import datetime

# Uses: MODEL_DIR, SAMPLES_DIR, FIG_DIR, EVAL_JSON, TASK4_OUT (defined above)

bundle_root = Path("/content") / f"task4_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
bundle_root.mkdir(parents=True, exist_ok=True)

# 1) pick checkpoint
ckpt_dir = Path(MODEL_DIR) / "checkpoints"
final_ckpt = ckpt_dir / "final.pt"
if final_ckpt.exists():
    ckpt_path = final_ckpt
else:
    step_ckpts = sorted(ckpt_dir.glob("step_*.pt"))
    ckpt_path = step_ckpts[-1] if step_ckpts else None

if ckpt_path is None:
    raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
print("Bundling checkpoint:", ckpt_path)

# 2) copy model metadata + checkpoint
(model_out := bundle_root / "best_xl_sp").mkdir(parents=True, exist_ok=True)
for name in ["config.json", "summary.json", "metrics.jsonl"]:
    src = Path(MODEL_DIR) / name
    if src.exists():
        shutil.copy2(src, model_out / name)
    else:
        print("Missing (ok):", src)

(model_ckpt_out := model_out / "checkpoints").mkdir(parents=True, exist_ok=True)
shutil.copy2(ckpt_path, model_ckpt_out / ckpt_path.name)

# 3) copy eval json if present
src_eval = Path(EVAL_JSON)
if src_eval.exists():
    shutil.copy2(src_eval, bundle_root / "eval_metrics.json")
else:
    print("No eval_metrics.json yet (ok):", src_eval)

# 4) copy samples / figures if present
for src_dir, dst_name in [(SAMPLES_DIR, "samples"), (FIG_DIR, "figures_report")]:
    src_dir = Path(src_dir)
    if src_dir.exists():
        shutil.copytree(src_dir, bundle_root / dst_name, dirs_exist_ok=True)
    else:
        print("Missing (ok):", src_dir)

# 5) write a tiny manifest
(bundle_root / "MANIFEST.txt").write_text(
    "\n".join(
        [
            f"MODEL_DIR={MODEL_DIR}",
            f"CHECKPOINT_BUNDLED={ckpt_path}",
            f"SAMPLES_DIR={SAMPLES_DIR}",
            f"FIG_DIR={FIG_DIR}",
            f"EVAL_JSON={EVAL_JSON}",
        ]
    )
    + "\n",
    encoding="utf-8",
)

# 6) zip it
zip_base = str(bundle_root)  # shutil.make_archive wants a str without .zip
zip_path = shutil.make_archive(zip_base, "zip", root_dir=bundle_root)
print("Created ZIP:", zip_path)

# 7) optional copy to Drive (may fail if quota is full)
try:
    drive_dst = Path(TASK4_OUT) / Path(zip_path).name
    shutil.copy2(zip_path, drive_dst)
    print("Copied ZIP to Drive:", drive_dst)
except Exception as e:
    print("Could not copy ZIP to Drive (quota/full is common). Download from /content instead.")
    print("Copy error:", repr(e))"""
        )
    )

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": cells,
    }
    out = root / "notebooks" / "task4_colab_best_model.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()

