"""
Train one decoder-only LM on Part-1 jsonl + SentencePiece (Part 2 scaling study).

Example (from repository root):
  python -m scripts.task2.train --train-jsonl data/processed/train.jsonl \\
    --val-jsonl data/processed/val.jsonl --spm-model data/processed/spm.model \\
    --preset tiny --tokens-per-batch 32768 --block-size 512 --lr 3e-4 --out-dir outputs/task2/tiny

Requires: torch, sentencepiece, numpy
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.task2.config_presets import PRESETS
from scripts.task2.data import estimate_val_loss, get_batch, load_jsonl_token_stream
from scripts.task2.model import SVGTransformerLM


def _cosine_lr(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    t = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--spm-model", type=Path, required=True)
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="tiny")
    p.add_argument("--block-size", type=int, default=512, help="Sequence length T; tokens/step = batch*T.")
    p.add_argument(
        "--tokens-per-batch",
        type=int,
        default=32768,
        help="Keep constant across model sizes (assignment): batch_size = tokens_per_batch // block_size.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5, help="Cosine floor.")
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=1, help="For scaling study use 1.")
    p.add_argument("--max-docs", type=int, default=None, help="Debug: cap number of jsonl rows.")
    p.add_argument("--val-batches", type=int, default=40, help="Batches for val loss estimate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "task2" / "run")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.device == "cuda":
        print("CUDA not available; using CPU (slow).")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    preset = PRESETS[args.preset]
    block_size = args.block_size
    tokens_per_batch = args.tokens_per_batch
    if tokens_per_batch % block_size != 0:
        raise SystemExit("tokens-per-batch must be divisible by block-size")
    batch_size = tokens_per_batch // block_size

    print("Loading train stream ...")
    train_data, vocab_size = load_jsonl_token_stream(
        args.train_jsonl, args.spm_model, max_docs=args.max_docs
    )
    print("Loading val stream ...")
    val_data, vs2 = load_jsonl_token_stream(args.val_jsonl, args.spm_model, max_docs=args.max_docs)
    if vs2 != vocab_size:
        raise RuntimeError("Vocab mismatch train/val")

    print(f"Train tokens (incl. EOS): {len(train_data):,} | Val: {len(val_data):,}")

    model = SVGTransformerLM(
        vocab_size=vocab_size,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        d_ff=preset.d_ff,
        max_seq_len=block_size,
        dropout=args.dropout,
    ).to(device)
    n_params = model.count_parameters()
    print(f"Preset={preset.name} ~{preset.approx_params} | Actual trainable params: {n_params:,}")

    steps_per_epoch = max(1, (len(train_data) - 1) // tokens_per_batch)
    max_steps = steps_per_epoch * args.epochs
    print(f"batch_size={batch_size} block_size={block_size} tokens/step={tokens_per_batch}")
    print(f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} -> max_steps={max_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    config_path = args.out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "preset": preset.name,
                "approx_params": preset.approx_params,
                "n_params": n_params,
                "d_model": preset.d_model,
                "n_layers": preset.n_layers,
                "n_heads": preset.n_heads,
                "d_ff": preset.d_ff,
                "vocab_size": vocab_size,
                "block_size": block_size,
                "tokens_per_batch": tokens_per_batch,
                "batch_size": batch_size,
                "lr": args.lr,
                "min_lr": args.min_lr,
                "warmup_steps": args.warmup_steps,
                "epochs": args.epochs,
                "max_steps": max_steps,
                "train_tokens": len(train_data),
                "val_tokens": len(val_data),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path.write_text("", encoding="utf-8")
    model.train()
    t0 = time.perf_counter()
    step = 0

    while step < max_steps:
        lr = _cosine_lr(
            step,
            warmup_steps=args.warmup_steps,
            max_steps=max_steps,
            max_lr=args.lr,
            min_lr=args.min_lr,
        )
        for pg in opt.param_groups:
            pg["lr"] = lr

        t_step = time.perf_counter()
        x, y = get_batch(train_data, batch_size, block_size, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        dt = time.perf_counter() - t_step
        tok_s = tokens_per_batch / max(dt, 1e-8)
        mem_gb = 0.0
        if device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

        rec = {
            "step": step,
            "train_loss": loss.item(),
            "lr": lr,
            "tokens_per_s": tok_s,
            "wall_time_s": time.perf_counter() - t0,
            "gpu_mem_gb": mem_gb,
        }
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec) + "\n")

        if step % 50 == 0 or step == max_steps - 1:
            print(
                f"step {step}/{max_steps} train_loss={loss.item():.4f} lr={lr:.2e} tok/s={tok_s:.0f}",
                flush=True,
            )

        step += 1

    wall = time.perf_counter() - t0
    print("Evaluating validation loss ...")
    val_rng = np.random.default_rng(args.seed + 1)
    val_loss = estimate_val_loss(
        model, val_data, batch_size, block_size, device, val_rng, args.val_batches
    )

    summary = {
        "preset": preset.name,
        "n_params": n_params,
        "val_loss": val_loss,
        "wall_time_epoch_s": wall,
        "steps": max_steps,
        "train_jsonl": str(args.train_jsonl),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. val_loss={val_loss:.4f} wall_time={wall:.1f}s | wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
