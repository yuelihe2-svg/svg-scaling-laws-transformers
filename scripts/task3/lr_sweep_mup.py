"""LR sweep on the smallest μP preset (default: tiny), same protocol as Part 2."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--spm-model", type=Path, required=True)
    p.add_argument("--lrs", type=float, nargs="+", required=True)
    p.add_argument("--preset", type=str, default="tiny", help="Smallest μP model for sweep.")
    p.add_argument("--tokens-per-batch", type=int, default=32768)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--out-csv", type=Path, default=REPO_ROOT / "outputs" / "task3" / "lr_sweep_mup.csv")
    args = p.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for lr in args.lrs:
        out = args.out_csv.parent / f"sweep_mup_{args.preset}_lr{lr}"
        cmd = [
            sys.executable,
            "-m",
            "scripts.task3.train_mup",
            "--train-jsonl",
            str(args.train_jsonl),
            "--val-jsonl",
            str(args.val_jsonl),
            "--spm-model",
            str(args.spm_model),
            "--preset",
            str(args.preset),
            "--lr",
            str(lr),
            "--tokens-per-batch",
            str(args.tokens_per_batch),
            "--block-size",
            str(args.block_size),
            "--warmup-steps",
            str(args.warmup_steps),
            "--out-dir",
            str(out),
        ]
        if args.max_docs is not None:
            cmd.extend(["--max-docs", str(args.max_docs)])
        print("Running:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        rows.append({"lr": lr, "val_loss": summary["val_loss"], "n_params": summary["n_params"]})

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lr", "val_loss", "n_params"])
        w.writeheader()
        w.writerows(rows)

    best = min(rows, key=lambda x: x["val_loss"])
    print(f"Best lr={best['lr']} val_loss={best['val_loss']}")
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
