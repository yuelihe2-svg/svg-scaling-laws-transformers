"""
Check that cleaned SVGs rasterize with CairoSVG (same gate as preprocess --render-check).

Does not rewrite data; samples lines from jsonl and reports pass/fail counts.

From repo root:
  python scripts/validate_render.py --jsonl data/processed/train.jsonl --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svg_scaling.cleaning import cairosvg_available, try_render_svg  # noqa: E402


def _reservoir_sample_lines(path: Path, k: int, rng: random.Random) -> list[str]:
    """One-pass reservoir sample without loading the whole file."""
    reservoir: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            if len(reservoir) < k:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = line
    return reservoir


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True, help="e.g. data/processed/train.jsonl")
    p.add_argument("--max-samples", type=int, default=500, help="Random sample size (after shuffle).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ok, msg = cairosvg_available()
    if not ok:
        print("CairoSVG / Cairo not usable:", msg)
        print(
            "On Windows: install a Cairo runtime (e.g. GTK3 runtime or MSYS2 cairo) and ensure "
            "its bin folder is on PATH, then: pip install cairosvg",
        )
        return 1

    rng = random.Random(args.seed)
    lines = _reservoir_sample_lines(args.jsonl, args.max_samples, rng)
    if not lines:
        print("Empty jsonl")
        return 1

    passed = 0
    failed = 0
    for line in lines:
        row = json.loads(line)
        svg = row.get("svg", "")
        if try_render_svg(svg):
            passed += 1
        else:
            failed += 1

    n = passed + failed
    pct = 100.0 * passed / n if n else 0.0
    print(f"CairoSVG rasterization: {passed}/{n} passed ({pct:.1f}%), {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
