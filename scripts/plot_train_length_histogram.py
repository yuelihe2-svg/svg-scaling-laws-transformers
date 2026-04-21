#!/usr/bin/env python3
"""Plot train_length_histogram from data/processed/stats.json (PNG + PDF)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stats",
        type=Path,
        default=Path("data/processed/stats.json"),
        help="Path to stats.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/figures/train_length_histogram"),
        help="Output path without extension; writes .png and .pdf",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="PNG resolution",
    )
    args = p.parse_args()

    with args.stats.open(encoding="utf-8") as f:
        stats = json.load(f)

    hist = stats.get("train_length_histogram")
    if not hist:
        raise SystemExit(f"No train_length_histogram in {args.stats}")

    edges = np.asarray(hist["bin_edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    quantiles = {float(k): float(v) for k, v in hist.get("quantiles", {}).items()}

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax.stairs(counts, edges, fill=True, color="#4C78A8", alpha=0.85, label="Count per bin")
    ax.set_xlabel("Sequence length (num_tokens)")
    ax.set_ylabel("Number of training examples")
    ax.set_title("Training set token-length distribution")

    ymax = float(counts.max()) if len(counts) else 1.0
    q_styles = [
        (0.5, "C1", "median"),
        (0.9, "C2", "p90"),
        (0.99, "C3", "p99"),
    ]
    for q, color, name in q_styles:
        if q not in quantiles:
            continue
        x = quantiles[q]
        ax.axvline(x, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label=f"{name} = {x:.0f}")

    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, ymax * 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    png = args.out.with_suffix(".png")
    pdf = args.out.with_suffix(".pdf")
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png.resolve()}")
    print(f"Wrote {pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
