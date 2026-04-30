"""
Part 4: Prefix completion analysis figure.

Builds a side-by-side figure showing:
  (1) prefix text
  (2) model completion SVG snippet
  (3) rendered PNG (if available), otherwise a placeholder

Example:
  python -m scripts.task4.prefix_completion_figure \
    --samples-dir outputs/task4_final/samples \
    --out-dir outputs/task4_final/figures_report \
    --num-examples 3
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def _read_svg_snippet(path: Path, *, max_chars: int = 420) -> str:
    txt = path.read_text(encoding="utf-8", errors="ignore").strip()
    if len(txt) > max_chars:
        return txt[: max_chars - 3] + "..."
    return txt


def _load_img(path: Path) -> np.ndarray | None:
    try:
        return plt.imread(str(path))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-examples", type=int, default=3)
    ap.add_argument("--prefix-wrap", type=int, default=58)
    ap.add_argument("--snippet-wrap", type=int, default=68)
    args = ap.parse_args()

    manifest = json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
    prefix_recs = [r for r in manifest if r.get("type") == "prefix"]
    if not prefix_recs:
        raise SystemExit("No prefix samples found in manifest.json")

    # Prefer render_ok=true examples first, but include failures if needed to hit num_examples.
    prefix_recs = sorted(prefix_recs, key=lambda r: (not bool(r.get("render_ok", False)), int(r.get("i", 0))))
    prefix_recs = prefix_recs[: max(1, args.num_examples)]

    n = len(prefix_recs)
    fig = plt.figure(figsize=(14, 3.2 * n))
    gs = fig.add_gridspec(nrows=n, ncols=3, width_ratios=[1.05, 1.2, 1.0], wspace=0.25, hspace=0.35)

    for row, rec in enumerate(prefix_recs):
        prefix = str(rec.get("prefix", "")).strip()

        svg_name = Path(str(rec["svg_path"])).name
        png_name = Path(str(rec.get("png_path", ""))).name if rec.get("png_path") else ""
        svg_path = args.samples_dir / "svg" / svg_name
        png_path = args.samples_dir / "png" / png_name if png_name else None

        completion = _read_svg_snippet(svg_path, max_chars=520)
        render_ok = bool(rec.get("render_ok", False))

        # 1) Prefix text
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.axis("off")
        ax0.set_title(f"Prefix (example {int(rec.get('i', row))})", fontsize=11, pad=8)
        ax0.text(
            0,
            1,
            _wrap(prefix, args.prefix_wrap),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9,
        )

        # 2) Completion snippet
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.axis("off")
        ax1.set_title("Model completion (SVG snippet)", fontsize=11, pad=8)
        ax1.text(
            0,
            1,
            _wrap(completion, args.snippet_wrap),
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

        # 3) Rendered image (or placeholder)
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.axis("off")
        ax2.set_title("Rendered PNG" + (" (ok)" if render_ok else " (failed)"), fontsize=11, pad=8)
        img = _load_img(png_path) if (png_path is not None and png_path.exists()) else None
        if img is None:
            # placeholder
            ph = np.ones((220, 220, 3), dtype=np.float32)
            ph[:] = np.array([0.95, 0.95, 0.95], dtype=np.float32)
            ax2.imshow(ph)
            ax2.text(
                0.5,
                0.5,
                "Render failed\n(or PNG missing)",
                ha="center",
                va="center",
                fontsize=11,
                color="black",
            )
        else:
            ax2.imshow(img)

        ax2.text(
            0.5,
            -0.03,
            f"temp={rec.get('temperature')}  top-k={rec.get('top_k')}  top-p={rec.get('top_p')}",
            ha="center",
            va="top",
            transform=ax2.transAxes,
            fontsize=9,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_png = args.out_dir / "prefix_completion_examples.png"
    out_pdf = args.out_dir / "prefix_completion_examples.pdf"
    fig.suptitle("Prefix completion analysis (prefix → completion → render)", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    print("Wrote", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

