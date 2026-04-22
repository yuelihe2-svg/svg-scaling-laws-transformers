"""
Generate Part 3 deliverable figures (μP scaling & extrapolation).

From repo root:
  python -m scripts.task3.figure_report --task2-dir outputs/task2 --task3-dir outputs/task3 --out-dir outputs/task3/figures_report

Writes:
  - lr_sweep_sp_vs_mup.png/.pdf
  - sp_vs_mup_scaling.png/.pdf
  - sp_fit.txt, mup_fit.txt
  - extrapolation_10x_xl.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def _load_metrics_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fit_power_law(n: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Fit L = a * N^{-alpha} + c via nonlinear least squares on log(N).
    Returns (theta=[a, alpha, c], rmse).
    """
    n = np.asarray(n, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    x = np.log(n)
    y = loss

    def residual(theta: np.ndarray) -> np.ndarray:
        a, alpha, c = theta
        return a * np.exp(-alpha * x) + c - y

    a0 = max(float(y.max() - y.min()), 1e-6)
    theta0 = np.array([a0, 0.3, float(y.min())], dtype=np.float64)
    bounds = ([0.0, 0.01, -np.inf], [np.inf, 3.0, np.inf])
    r = least_squares(residual, theta0, bounds=bounds, max_nfev=4000)
    theta = r.x.astype(np.float64)
    rmse = float(np.sqrt(np.mean(r.fun**2)))
    return theta, rmse


def _fit_covariance(n: np.ndarray, loss: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Approximate covariance of theta from J (Gauss-Newton) and residual variance.
    Returns 3x3 covariance; may be ill-conditioned for 5 points (use with care).
    """
    n = np.asarray(n, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    x = np.log(n)
    y = loss
    a, alpha, c = theta
    # residuals at solution
    yhat = a * np.exp(-alpha * x) + c
    r = yhat - y
    dof = max(1, len(y) - len(theta))
    s2 = float(np.sum(r**2) / dof)
    # Jacobian: d/da = exp(-alpha x), d/dalpha = -a*x*exp(-alpha x), d/dc = 1
    e = np.exp(-alpha * x)
    J = np.stack([e, -a * x * e, np.ones_like(x)], axis=1)  # (m,3)
    JTJ = J.T @ J
    try:
        cov = s2 * np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan, dtype=np.float64)
    return cov


def _predict(theta: np.ndarray, n: float) -> float:
    a, alpha, c = (float(theta[0]), float(theta[1]), float(theta[2]))
    return a * (float(n) ** (-alpha)) + c


def _predict_vec(theta: np.ndarray, n: np.ndarray) -> np.ndarray:
    a, alpha, c = (float(theta[0]), float(theta[1]), float(theta[2]))
    n = np.asarray(n, dtype=np.float64)
    return a * (n ** (-alpha)) + c


def _load_summaries(root: Path, presets: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names: list[str] = []
    ns: list[float] = []
    ls: list[float] = []
    for p in presets:
        s = json.loads((root / p / "summary.json").read_text(encoding="utf-8"))
        names.append(p)
        ns.append(float(s["n_params"]))
        ls.append(float(s["val_loss"]))
    return names, np.array(ns, dtype=np.float64), np.array(ls, dtype=np.float64)


def _read_lr_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lrs: list[float] = []
    losses: list[float] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lrs.append(float(row["lr"]))
            losses.append(float(row["val_loss"]))
    return np.array(lrs, dtype=np.float64), np.array(losses, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task2-dir", type=Path, default=Path("outputs/task2"))
    ap.add_argument("--task3-dir", type=Path, default=Path("outputs/task3"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/task3/figures_report"))
    ap.add_argument("--presets", nargs="+", default=["tiny", "small", "medium", "large", "xl"])
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # --- LR sweep compare (SP vs μP) ---
    sp_lr_csv = args.task2_dir / "lr_sweep.csv"
    mup_lr_csv = args.task3_dir / "lr_sweep_mup.csv"
    sp_lr, sp_l = _read_lr_csv(sp_lr_csv)
    mup_lr, mup_l = _read_lr_csv(mup_lr_csv)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(sp_lr, sp_l, marker="o", label="SP sweep (Part 2 Tiny)")
    ax.plot(mup_lr, mup_l, marker="s", label="μP sweep (Part 3 Tiny μP)")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("Learning-rate sweep comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "lr_sweep_sp_vs_mup.png", dpi=160)
    fig.savefig(out / "lr_sweep_sp_vs_mup.pdf")
    plt.close(fig)

    # --- scaling compare ---
    names, n_sp, l_sp = _load_summaries(args.task2_dir, args.presets)
    _, n_mup, l_mup = _load_summaries(args.task3_dir, args.presets)
    theta_sp, rmse_sp = _fit_power_law(n_sp, l_sp)
    theta_mup, rmse_mup = _fit_power_law(n_mup, l_mup)

    (out / "sp_fit.txt").write_text(
        "Fit (SP): L = a * N^(-alpha) + c\n"
        + f"a = {theta_sp[0]:.6g}\nalpha = {theta_sp[1]:.6f}\nc = {theta_sp[2]:.6f}\nrmse = {rmse_sp:.6f}\n",
        encoding="utf-8",
    )
    (out / "mup_fit.txt").write_text(
        "Fit (μP): L = a * N^(-alpha) + c\n"
        + f"a = {theta_mup[0]:.6g}\nalpha = {theta_mup[1]:.6f}\nc = {theta_mup[2]:.6f}\nrmse = {rmse_mup:.6f}\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(n_sp, l_sp, s=80, marker="o", label=f"SP (α={theta_sp[1]:.3f})")
    ax.scatter(n_mup, l_mup, s=80, marker="s", label=f"μP (α={theta_mup[1]:.3f})")
    for i, nm in enumerate(names):
        ax.annotate(nm, (n_sp[i], l_sp[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    xs = np.logspace(np.log10(min(n_sp.min(), n_mup.min())), np.log10(max(n_sp.max(), n_mup.max())), 160)
    ax.plot(xs, _predict_vec(theta_sp, xs), "--", color="C0", alpha=0.8)
    ax.plot(xs, _predict_vec(theta_mup, xs), "--", color="C1", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters N (log scale)")
    ax.set_ylabel("Validation loss (1 epoch)")
    ax.set_title("Scaling curves: standard parametrization vs μP")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sp_vs_mup_scaling.png", dpi=160)
    fig.savefig(out / "sp_vs_mup_scaling.pdf")
    plt.close(fig)

    # --- extrapolation: 10x XL params ---
    N_xl_sp = float(n_sp[-1])
    N_xl_mup = float(n_mup[-1])
    N_xl = max(N_xl_sp, N_xl_mup)
    N_pred = 10.0 * N_xl
    use = "sp" if rmse_sp <= rmse_mup else "mup"
    theta = theta_sp if use == "sp" else theta_mup
    rmse = rmse_sp if use == "sp" else rmse_mup
    cov = _fit_covariance(n_sp if use == "sp" else n_mup, l_sp if use == "sp" else l_mup, theta)

    L_pred = _predict(theta, N_pred)
    # delta-method variance of prediction: grad^T cov grad
    a, alpha, c = (float(theta[0]), float(theta[1]), float(theta[2]))
    # y = a * N^{-alpha} + c
    g = np.array([N_pred ** (-alpha), -a * (N_pred ** (-alpha)) * math.log(N_pred), 1.0], dtype=np.float64)
    var = float(g.T @ cov @ g) if np.isfinite(cov).all() else float("nan")
    se = math.sqrt(var) if var == var and var >= 0 else float("nan")
    lo = L_pred - 1.96 * se if se == se else float("nan")
    hi = L_pred + 1.96 * se if se == se else float("nan")

    (out / "extrapolation_10x_xl.txt").write_text(
        "Extrapolation using the better RMSE fit (lower is better).\n"
        f"chosen_curve = {use}\n"
        f"XL_N = {N_xl:.0f}\n"
        f"N_pred = 10 * XL_N = {N_pred:.0f}\n"
        f"predicted_val_loss = {L_pred:.6f}\n"
        f"approx_95pct_interval = [{lo:.6f}, {hi:.6f}]  (delta-method; small-n warning)\n"
        f"fit_rmse = {rmse:.6f}\n",
        encoding="utf-8",
    )

    # --- μP summary table (like Task 2) ---
    headers = ["preset", "n_params", "val_loss", "wall_s", "mean_tok/s", "peak_GB", "lr"]
    table_rows: list[list[str]] = []
    for preset in args.presets:
        summ = json.loads((args.task3_dir / preset / "summary.json").read_text(encoding="utf-8"))
        cfg_path = args.task3_dir / preset / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        metrics = _load_metrics_jsonl(args.task3_dir / preset / "metrics.jsonl")
        warm = 200
        tail = metrics[warm:] if len(metrics) > warm else metrics
        if tail:
            mean_tok = float(np.mean([float(r.get("tokens_per_s", 0.0)) for r in tail]))
        else:
            mean_tok = float("nan")
        peak_gb = (
            float(max((float(r.get("gpu_mem_gb", 0.0)) for r in metrics), default=float("nan")))
            if metrics
            else float("nan")
        )
        table_rows.append(
            [
                preset,
                f"{int(summ['n_params'])}",
                f"{float(summ['val_loss']):.4f}",
                f"{float(summ.get('wall_time_epoch_s', float('nan'))):.1f}",
                f"{mean_tok:.0f}" if mean_tok == mean_tok else "nan",
                f"{peak_gb:.2f}" if peak_gb == peak_gb else "nan",
                f"{float(cfg.get('lr', float('nan'))):g}" if cfg.get("lr") is not None else "nan",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 2.3))
    ax.axis("off")
    tbl = ax.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.scale(1, 1.4)
    fig.savefig(out / "mup_summary_table.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "mup_summary_table.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote Part 3 figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

