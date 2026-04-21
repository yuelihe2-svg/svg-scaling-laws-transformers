"""Load Part-1 jsonl + SentencePiece and build a flat token tensor for LM training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_jsonl_token_stream(
    jsonl_path: Path,
    sp_model_path: Path,
    *,
    max_docs: int | None = None,
) -> tuple[torch.Tensor, int]:
    """
    Concatenate all training texts into one 1D int64 tensor, separated by EOS.

    Returns (token_ids, vocab_size_from_processor).
    """
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    vocab_size = processor.get_piece_size()
    eos_id = processor.eos_id()
    if eos_id < 0:
        eos_id = 0

    chunks: list[list[int]] = []
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_docs is not None and n >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("svg", "")
            ids = processor.encode(text, out_type=int)
            if not ids:
                continue
            chunks.append(ids + [eos_id])
            n += 1

    if not chunks:
        raise RuntimeError(f"No sequences loaded from {jsonl_path}")

    flat = np.concatenate([np.asarray(c, dtype=np.int64) for c in chunks])
    return torch.from_numpy(flat), vocab_size


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random training batch: inputs (B,T), targets (B,T) next-token."""
    max_start = len(data) - block_size - 1
    if max_start < 1:
        raise ValueError("Token stream too short for block_size")
    ix = rng.integers(0, max_start, size=(batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_val_loss(
    model: torch.nn.Module,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    rng: np.random.Generator,
    num_batches: int,
) -> float:
    """Average cross-entropy on random val batches."""
    model.eval()
    losses: list[float] = []
    import torch.nn.functional as F

    max_start = len(val_data) - block_size - 1
    if max_start < 1:
        return float("nan")
    for _ in range(num_batches):
        ix = rng.integers(0, max_start, size=(batch_size,))
        x = torch.stack([val_data[i : i + block_size] for i in ix]).to(device)
        y = torch.stack([val_data[i + 1 : i + 1 + block_size] for i in ix]).to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))
