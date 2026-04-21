"""
Build Part-2 report figures from outputs under outputs/task2/.

From repo root:
  python -m scripts.task2.figure_report --task2-dir outputs/task2 --out-dir outputs/task2/figures_report

Writes:
  - points.json
  - scaling.png / scaling.pdf  (log N vs val_loss + power-law fit)
  - train_loss_all.png
  - bars_time_throughput_memory.png
  - summary_table.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fit_power_law(n: np.ndarray, loss: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.optimize import least_squares

    n = np.asarray(n, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    x = np.log(n)
    y = loss

    def residual(theta: np.ndarray) -> np.ndarray:
        a, alpha, c = theta
        return a * np.exp(-alpha * x) + c - y

    a0 = max(float(y.max() - y.min()), 1e-6)
    theta0 = np.array([a0, 0.3, float(y.min())])
    bounds = ([0.0, 0.01, -np.inf], [np.inf, 3.0, np.inf])
    r = least_squares(residual, theta0, bounds=bounds, max_nfev=2000)
    a, alpha, c = r.x
    rmse = float(np.sqrt(np.mean(r.fun**2)))
    return float(a), float(alpha), float(c), rmse


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task2-dir", type=Path, default=Path("outputs/task2"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/task2/figures_report"))
    p.add_argument(
        "--presets",
        nargs="+",
        default=["tiny", "small", "medium", "large", "xl"],
        help="Preset folder names under task2-dir (must have summary.json + metrics.jsonl).",
    )
    args = p.parse_args()

    root: Path = args.task2_dir
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    rows_table: list[list[str]] = []

    for preset in args.presets:
        s_path = root / preset / "summary.json"
        c_path = root / preset / "config.json"
        m_path = root / preset / "metrics.jsonl"
        if not s_path.exists():
            raise SystemExit(f"Missing {s_path}")
        s = json.loads(s_path.read_text(encoding="utf-8"))
        c = json.loads(c_path.read_text(encoding="utf-8")) if c_path.exists() else {}
        metrics = _load_metrics(m_path) if m_path.exists() else []

        warm = 200
        tail = metrics[warm:] if len(metrics) > warm else metrics
        mean_tok = statistics.mean(float(r["tokens_per_s"]) for r in tail) if tail else float("nan")
        max_mem = max((float(r.get("gpu_mem_gb", 0.0)) for r in metrics), default=float("nan"))

        points.append(
            {
                "preset": preset,
                "n_params": int(s["n_params"]),
                "val_loss": float(s["val_loss"]),
            }
        )
        rows_table.append(
            [
                preset,
                str(s["n_params"]),
                f"{s['val_loss']:.4f}",
                f"{s.get('wall_time_epoch_s', 0):.1f}",
                f"{mean_tok:.0f}" if mean_tok == mean_tok else "nan",
                f"{max_mem:.2f}" if max_mem == max_mem else "nan",
                str(c.get("tokens_per_batch", "")),
                str(c.get("block_size", "")),
                str(c.get("lr", "")),
            ]
        )

    (out / "points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")

    n = np.array([p["n_params"] for p in points], dtype=np.float64)
    loss = np.array([p["val_loss"] for p in points], dtype=np.float64)
    a, alpha, c, rmse = _fit_power_law(n, loss)

    # --- scaling ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(n, loss, s=90, zorder=3)
    for p in points:
        ax.annotate(p["preset"], (p["n_params"], p["val_loss"]), xytext=(5, 5), textcoords="offset points")
    xs = np.logspace(np.log10(n.min()), np.log10(n.max()), 120)
    ys = a * xs ** (-alpha) + c
    fit_label = r"$L=aN^{-\alpha}+c$" + f",  α={alpha:.3f}, RMSE={rmse:.4f}"
    ax.plot(xs, ys, "k--", lw=1.5, alpha=0.75, label=fit_label)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N (log scale)")
    ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("Scaling law (SVG LM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scaling.png", dpi=160)
    fig.savefig(out / "scaling.pdf")
    plt.close(fig)

    # --- train loss curves (one figure) ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, preset in enumerate(args.presets):
        ax = axes[i]
        m_path = root / preset / "metrics.jsonl"
        m = _load_metrics(m_path)
        steps = [int(r["step"]) for r in m]
        losses = [float(r["train_loss"]) for r in m]
        ax.plot(steps, losses, lw=0.8)
        ax.set_title(preset)
        ax.set_xlabel("step")
        ax.set_ylabel("train loss")
        ax.grid(True, alpha=0.3)
    for j in range(len(args.presets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Training loss vs step (1 epoch)", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "train_loss_all.png", dpi=160)
    fig.savefig(out / "train_loss_all.pdf")
    plt.close(fig)

    # --- bars: wall time, mean tok/s, max mem ---
    names = [p["preset"] for p in points]
    walls = [float(json.loads((root / pr / "summary.json").read_text(encoding="utf-8"))["wall_time_epoch_s"]) for pr in names]
    mean_toks: list[float] = []
    max_mems: list[float] = []
    for pr in names:
        m = _load_metrics(root / pr / "metrics.jsonl")
        warm = 200
        tail = m[warm:] if len(m) > warm else m
        mean_toks.append(statistics.mean(float(r["tokens_per_s"]) for r in tail) if tail else float("nan"))
        max_mems.append(max((float(r.get("gpu_mem_gb", 0.0)) for r in m), default=float("nan")))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(names, walls, color="steelblue")
    axes[0].set_title("Wall time / epoch (s)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, mean_toks, color="seagreen")
    axes[1].set_title("Mean throughput (tok/s)\n(after step 200)")
    axes[1].tick_params(axis="x", rotation=20)
    axes[2].bar(names, max_mems, color="coral")
    axes[2].set_title("Peak GPU memory (GB)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "bars_time_throughput_memory.png", dpi=160)
    fig.savefig(out / "bars_time_throughput_memory.pdf")
    plt.close(fig)

    # --- summary table image ---
    headers = ["preset", "n_params", "val_loss", "wall_s", "mean_tok/s", "peak_GB", "tok/batch", "block", "lr"]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows_table,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1, 1.4)
    fig.savefig(out / "summary_table.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "summary_table.pdf", bbox_inches="tight")
    plt.close(fig)

    fit_txt = out / "scaling_fit.txt"
    fit_txt.write_text(
        f"Fit: L = a * N^(-alpha) + c\n"
        f"a = {a:.6g}\n"
        f"alpha = {alpha:.6f}\n"
        f"c = {c:.6f}\n"
        f"rmse = {rmse:.6f}\n",
        encoding="utf-8",
    )

    print(f"Wrote figures under {out}")
    print(fit_txt.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
