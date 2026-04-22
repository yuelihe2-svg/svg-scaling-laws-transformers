"""
Part 4: Train the chosen best model longer and save checkpoints.

Default choice: **SP XL** (same model code as Part 2) with the Part 2 LR strategy.

From repo root (example):
  python -m scripts.task4.train_best_model \
    --train-jsonl data/processed/train.jsonl --val-jsonl data/processed/val.jsonl --spm-model data/processed/spm.model \
    --preset xl --lr 0.01 --tokens-per-batch 32768 --block-size 512 \
    --epochs 3 --save-every-steps 500 --out-dir outputs/task4/best_xl_sp

Writes:
  - config.json
  - metrics.jsonl
  - checkpoints/step_XXXXX.pt
  - summary.json (final val loss, wall time)
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


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    step: int,
    rng_state: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "rng_state": rng_state,
        },
        path,
    )


def _load_checkpoint(path: Path, *, model: torch.nn.Module, opt: torch.optim.Optimizer) -> tuple[int, dict]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    return int(ckpt["step"]), dict(ckpt.get("rng_state", {}))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", type=Path, required=True)
    p.add_argument("--val-jsonl", type=Path, required=True)
    p.add_argument("--spm-model", type=Path, required=True)
    p.add_argument("--preset", type=str, choices=list(PRESETS.keys()), default="xl")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--tokens-per-batch", type=int, default=32768)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--val-batches", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--save-every-steps", type=int, default=500)
    p.add_argument("--resume", type=Path, default=None, help="Path to a checkpoint .pt to resume from.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "task4" / "best_xl_sp")
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
    train_data, vocab_size = load_jsonl_token_stream(args.train_jsonl, args.spm_model)
    print("Loading val stream ...")
    val_data, vs2 = load_jsonl_token_stream(args.val_jsonl, args.spm_model)
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
    print(f"Preset={preset.name} ~{preset.approx_params} | trainable params: {n_params:,}")

    steps_per_epoch = max(1, (len(train_data) - 1) // tokens_per_batch)
    max_steps = steps_per_epoch * args.epochs
    print(f"batch_size={batch_size} block_size={block_size} tokens/step={tokens_per_batch}")
    print(f"steps_per_epoch={steps_per_epoch} epochs={args.epochs} -> max_steps={max_steps}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.jsonl"
    config_path = args.out_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "part": 4,
                "parameterization": "sp",
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
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start_step = 0
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        start_step, rs = _load_checkpoint(args.resume, model=model, opt=opt)
        if "np_rng_state" in rs:
            rng.bit_generator.state = rs["np_rng_state"]
        if "torch_rng_state" in rs:
            torch.set_rng_state(rs["torch_rng_state"])
        print("Resumed at step", start_step)

    if start_step == 0:
        metrics_path.write_text("", encoding="utf-8")

    model.train()
    t0 = time.perf_counter()
    step = start_step

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
            "train_loss": float(loss.item()),
            "lr": float(lr),
            "tokens_per_s": float(tok_s),
            "wall_time_s": float(time.perf_counter() - t0),
            "gpu_mem_gb": float(mem_gb),
        }
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(rec) + "\n")

        if step % 50 == 0 or step == max_steps - 1:
            print(
                f"step {step}/{max_steps} train_loss={loss.item():.4f} lr={lr:.2e} tok/s={tok_s:.0f}",
                flush=True,
            )

        if args.save_every_steps > 0 and (step % args.save_every_steps == 0) and step != start_step:
            ckpt_path = args.out_dir / "checkpoints" / f"step_{step:06d}.pt"
            _save_checkpoint(
                ckpt_path,
                model=model,
                opt=opt,
                step=step,
                rng_state={
                    "np_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                },
            )

        step += 1

    wall = time.perf_counter() - t0
    print("Evaluating validation loss ...")
    val_rng = np.random.default_rng(args.seed + 1)
    val_loss = estimate_val_loss(model, val_data, batch_size, block_size, device, val_rng, args.val_batches)

    # final checkpoint
    _save_checkpoint(
        args.out_dir / "checkpoints" / "final.pt",
        model=model,
        opt=opt,
        step=max_steps,
        rng_state={
            "np_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        },
    )

    summary = {
        "part": 4,
        "parameterization": "sp",
        "preset": preset.name,
        "n_params": n_params,
        "val_loss": float(val_loss),
        "wall_time_s": float(wall),
        "steps": int(max_steps),
        "train_jsonl": str(args.train_jsonl),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. val_loss={val_loss:.4f} wall_time={wall:.1f}s | wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

