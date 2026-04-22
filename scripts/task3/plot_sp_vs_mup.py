"""Overlay Part 2 (SP) and Part 3 (μP) scaling curves + optional power-law fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def _fit(n: np.ndarray, loss: np.ndarray) -> tuple[float, float, float, float]:
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


def load_points(task_dir: Path, presets: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    ns: list[float] = []
    ls: list[float] = []
    for pr in presets:
        s = json.loads((task_dir / pr / "summary.json").read_text(encoding="utf-8"))
        names.append(pr)
        ns.append(float(s["n_params"]))
        ls.append(float(s["val_loss"]))
    return names, np.array(ns), np.array(ls)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task2-dir", type=Path, default=Path("outputs/task2"))
    p.add_argument("--task3-dir", type=Path, default=Path("outputs/task3"))
    p.add_argument(
        "--presets",
        nargs="+",
        default=["tiny", "small", "medium", "large", "xl"],
    )
    p.add_argument("--out", type=Path, default=Path("outputs/task3/figures/sp_vs_mup_scaling.png"))
    args = p.parse_args()

    _, n2, l2 = load_points(args.task2_dir, args.presets)
    names, n3, l3 = load_points(args.task3_dir, args.presets)

    a2, al2, c2, _ = _fit(n2, l2)
    a3, al3, c3, _ = _fit(n3, l3)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n2, l2, s=80, marker="o", label="Part 2 (SP)")
    ax.scatter(n3, l3, s=80, marker="s", label="Part 3 (μP)")
    for i, name in enumerate(names):
        ax.annotate(name, (n2[i], l2[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    xs = np.logspace(np.log10(min(n2.min(), n3.min())), np.log10(max(n2.max(), n3.max())), 100)
    ax.plot(xs, a2 * xs ** (-al2) + c2, "--", color="C0", alpha=0.8, label=rf"SP fit $\alpha$={al2:.3f}")
    ax.plot(xs, a3 * xs ** (-al3) + c3, "--", color="C1", alpha=0.8, label=rf"μP fit $\alpha$={al3:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N (log)")
    ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("Standard parametrization vs μP")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
