"""
Fit L = a * N^(-alpha) + c to (params, val_loss) and plot.

  python -m scripts.task2.plot_scaling --results outputs/task2/summary_all.json --out outputs/task2/scaling.png

`summary_all.json` should be a JSON list of objects with keys n_params, val_loss, preset (optional).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def fit_power_law(n: np.ndarray, loss: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares fit of y = a * x^(-alpha) + c with alpha in (0, 2)."""
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
    return float(a), float(alpha), float(c), float(np.sqrt(np.mean(r.fun**2)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True, help="JSON list [{n_params, val_loss}, ...]")
    p.add_argument("--out", type=Path, default=Path("outputs/task2/scaling_plot.png"))
    args = p.parse_args()

    raw = json.loads(args.results.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "points" in raw:
        raw = raw["points"]
    n_list = [float(r["n_params"]) for r in raw]
    l_list = [float(r["val_loss"]) for r in raw]
    n = np.array(n_list)
    loss = np.array(l_list)

    try:
        a, alpha, c, rmse = fit_power_law(n, loss)
        fit_note = f"a={a:.4g}, alpha={alpha:.4f}, c={c:.4f}, rmse={rmse:.4f}"
    except ImportError:
        a, alpha, c, fit_note = 0.0, 0.0, 0.0, "install scipy for power-law fit: pip install scipy"

    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(n, loss, s=80, zorder=3)
    for i, r in enumerate(raw):
        name = r.get("preset", str(i))
        ax.annotate(name, (n_list[i], l_list[i]), textcoords="offset points", xytext=(4, 4))

    xs = np.logspace(np.log10(n.min()), np.log10(n.max()), 100)
    if alpha > 0:
        ys = a * xs ** (-alpha) + c
        ax.plot(xs, ys, "k--", alpha=0.7, label=fit_note)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N")
    ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("Scaling law (SVG LM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    print(fit_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
