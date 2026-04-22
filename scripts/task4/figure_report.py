"""
Part 4: build a rendered grid of samples for the report.

Expects:
  outputs/task4/samples/manifest.json and outputs/task4/samples/png/*.png

Example:
  python -m scripts.task4.figure_report --samples-dir outputs/task4/samples --out-dir outputs/task4/figures_report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_img(path: Path) -> np.ndarray:
    # Use matplotlib's reader to avoid extra hard deps beyond requirements.txt.
    return plt.imread(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, default=Path("outputs/task4/samples"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/task4/figures_report"))
    ap.add_argument("--max-images", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    manifest = json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
    pngs: list[Path] = []
    for rec in manifest:
        p = rec.get("png_path")
        if p and Path(p).exists():
            pngs.append(Path(p))
    pngs = pngs[: args.max_images]
    if not pngs:
        raise SystemExit("No PNGs found. Re-run sample_generate.py with --render.")

    cols = max(1, args.cols)
    rows = int(np.ceil(len(pngs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i >= len(pngs):
            continue
        img = _load_img(pngs[i])
        ax.imshow(img)
        ax.set_title(pngs[i].name, fontsize=8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "samples_grid.png"
    out_pdf = args.out_dir / "samples_grid.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Wrote", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

