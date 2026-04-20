"""
Smoke-test Hugging Face access and inspect starvector/svg-icons-simple.

Usage (from repo root):
  python scripts/verify_dataset.py
"""

from __future__ import annotations

import argparse
import sys

from datasets import Dataset, DatasetDict, load_dataset


DEFAULT_DATASET = "starvector/svg-icons-simple"
# Official dataset card uses capitalized column names.
SVG_KEYS = ("Svg", "svg", "SVG")
NAME_KEYS = ("Filename", "filename", "id", "name")


def _pick_field(row: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in row:
            return k
    return None


def describe(ds: DatasetDict | Dataset) -> None:
    if isinstance(ds, DatasetDict):
        print("Splits:", list(ds.keys()))
        for split, part in ds.items():
            print(f"  [{split}] rows={len(part)} features={part.features}")
    else:
        print(f"Single dataset: rows={len(ds)} features={ds.features}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"HuggingFace dataset id (default: {DEFAULT_DATASET})",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Optional split name to sample from (default: first available)",
    )
    args = p.parse_args()

    print(f"Loading {args.dataset!r} ...")
    ds = load_dataset(args.dataset)
    describe(ds)

    if isinstance(ds, DatasetDict):
        split = args.split or next(iter(ds.keys()))
        if split not in ds:
            print(f"Unknown split {split!r}. Available: {list(ds.keys())}", file=sys.stderr)
            return 1
        part = ds[split]
    else:
        split = "all"
        part = ds

    row0 = part[0]
    svg_key = _pick_field(row0, SVG_KEYS)
    name_key = _pick_field(row0, NAME_KEYS)
    if not svg_key:
        print("Could not find an SVG text column. First row keys:", list(row0.keys()), file=sys.stderr)
        return 1

    svg = row0[svg_key]
    label = row0.get(name_key, "<no name field>") if name_key else "<no name field>"
    print()
    print(f"Example from split={split!r}")
    print(f"  name/id field ({name_key!r}): {label!r}")
    print(f"  svg field ({svg_key!r}) length: {len(svg)} chars")
    preview = svg[:500].replace("\n", "\\n")
    print(f"  svg preview (first 500 chars): {preview!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
