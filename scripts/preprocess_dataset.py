"""
Part 1 — Build cleaned SVG splits, train a SentencePiece BPE model, filter by token length, save stats.

Run from repository root:
  python scripts/preprocess_dataset.py --output-dir data/processed
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections import Counter
from numbers import Integral
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from svg_scaling.cleaning import cairosvg_available, clean_svg, try_render_svg  # noqa: E402

try:
    import sentencepiece as spm
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please install sentencepiece: pip install sentencepiece") from e


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _normalize_hf_filename(raw: object) -> int | str:
    """Icons/fonts use numeric ``Filename``; svg-emoji-simple uses string ids."""
    if isinstance(raw, bool):
        return str(raw)
    if isinstance(raw, Integral):
        return int(raw)
    return str(raw)


def _load_rows_from_hf(dataset_name: str, max_rows: int | None = None) -> list[dict]:
    """Load one HF dataset; dedupe by Filename within that dataset; tag source_dataset.

    If ``max_rows`` is set, only rows from the ``train`` split (or the first split if no ``train``)
    are loaded, up to ``max_rows`` rows — used to subsample very large sets like svg-fonts-simple.
    """
    ds: DatasetDict = load_dataset(dataset_name)
    split_names = list(ds.keys())
    if max_rows is not None:
        primary = "train" if "train" in ds else split_names[0]
        part0 = ds[primary]
        n = min(max_rows, len(part0))
        part0 = part0.select(range(n))
        split_names = [primary]
        ds = DatasetDict({primary: part0})
    by_id: dict[int | str, dict] = {}
    for split_name in split_names:
        part = ds[split_name]
        for row in tqdm(part, desc=f"load[{dataset_name}][{split_name}]"):
            fid = _normalize_hf_filename(row["Filename"])
            if fid not in by_id:
                by_id[fid] = {
                    "filename": fid,
                    "source_dataset": dataset_name,
                    "svg_raw": row["Svg"],
                    "source_split": split_name,
                }
    return list(by_id.values())


def _merge_hf_datasets(
    dataset_names: list[str],
    row_limits: dict[str, int] | None = None,
) -> list[dict]:
    """Merge several datasets; same (source_dataset, filename) only once."""
    row_limits = row_limits or {}
    seen: set[tuple[str, int | str]] = set()
    out: list[dict] = []
    for name in dataset_names:
        limit = row_limits.get(name)
        for row in _load_rows_from_hf(name, max_rows=limit):
            key = (row["source_dataset"], row["filename"])
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


def _clean_rows(
    rows: list[dict],
    min_chars: int,
    max_chars: int,
    render_check: bool,
) -> tuple[list[dict], Counter]:
    stats: Counter = Counter()
    out: list[dict] = []
    for row in tqdm(rows, desc="clean"):
        raw = row["svg_raw"]
        cleaned = clean_svg(raw)
        if cleaned is None:
            stats["drop_parse"] += 1
            continue
        if len(cleaned) < min_chars:
            stats["drop_short"] += 1
            continue
        if len(cleaned) > max_chars:
            stats["drop_long_chars"] += 1
            continue
        if render_check and not try_render_svg(cleaned):
            stats["drop_render"] += 1
            continue
        out.append(
            {
                "filename": row["filename"],
                "source_dataset": row["source_dataset"],
                "svg": cleaned,
                "source_split": row["source_split"],
            }
        )
    return out, stats


def _split_rows(
    rows: list[dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    n = len(rows)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]
    train = [rows[i] for i in train_idx]
    val = [rows[i] for i in val_idx]
    test = [rows[i] for i in test_idx]
    return train, val, test


def _write_jsonl(path: Path, rows: list[dict], keep_meta: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            obj = dict(r)
            if not keep_meta:
                obj.pop("source_split", None)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _train_sentencepiece(train_rows: list[dict], prefix_path: Path, vocab_size: int) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
        tmp_path = Path(tmp.name)
        for r in train_rows:
            line = r["svg"].replace("\n", " ").strip() + "\n"
            tmp.write(line)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stderr(devnull):
                spm.SentencePieceTrainer.train(
                    input=str(tmp_path),
                    model_prefix=str(prefix_path),
                    vocab_size=vocab_size,
                    model_type="bpe",
                    character_coverage=0.9995,
                    hard_vocab_limit=False,
                )
    finally:
        tmp_path.unlink(missing_ok=True)


def _encode_lengths(processor: spm.SentencePieceProcessor, text: str) -> int:
    return len(processor.encode(text, out_type=int))


def _filter_by_tokens(
    rows: list[dict],
    processor: spm.SentencePieceProcessor,
    max_token_len: int,
) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for r in tqdm(rows, desc="token-filter"):
        n = _encode_lengths(processor, r["svg"])
        if n > max_token_len:
            dropped += 1
            continue
        row = dict(r)
        row["num_tokens"] = n
        kept.append(row)
    return kept, dropped


def _total_tokens(rows: list[dict]) -> int:
    return int(sum(r["num_tokens"] for r in rows))


def _length_histogram(lengths: list[int], bins: int = 40) -> dict:
    if not lengths:
        return {"bins": [], "counts": [], "quantiles": {}}
    arr = np.asarray(lengths, dtype=np.int64)
    counts, edges = np.histogram(arr, bins=bins)
    qs = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
    quantiles = {str(q): float(np.quantile(arr, q)) for q in qs}
    return {
        "bin_edges": edges.tolist(),
        "counts": counts.tolist(),
        "quantiles": quantiles,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="starvector/svg-icons-simple",
        help="Primary HuggingFace dataset id.",
    )
    p.add_argument(
        "--extra-datasets",
        action="append",
        default=None,
        metavar="ID",
        help="Additional HF datasets to merge (repeat flag), e.g. --extra-datasets starvector/svg-emoji-simple",
    )
    p.add_argument(
        "--fonts-subsample",
        type=int,
        default=None,
        metavar="N",
        help="When starvector/svg-fonts-simple is included, load at most N rows from its train split "
        "(default: full dataset). Use this to cap memory/time while reaching token targets.",
    )
    p.add_argument("--output-dir", type=Path, default=_repo_root() / "data" / "processed")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--val-ratio", type=float, default=0.01)
    p.add_argument("--test-ratio", type=float, default=0.01)
    p.add_argument("--min-chars", type=int, default=50)
    p.add_argument("--max-chars", type=int, default=200_000)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--max-token-len", type=int, default=2048)
    p.add_argument("--render-check", action="store_true", help="Drop SVGs CairoSVG cannot render (slower).")
    p.add_argument("--keep-meta", action="store_true", help="Keep source_split in jsonl outputs.")
    p.add_argument("--min-train-tokens", type=int, default=100_000_000)
    args = p.parse_args()

    if args.render_check:
        ok, msg = cairosvg_available()
        if not ok:
            raise SystemExit(
                f"--render-check needs CairoSVG with native Cairo: {msg}\n"
                "Install Cairo (e.g. GTK for Windows runtime or MSYS2 mingw-w64-cairo) and ensure "
                "DLLs are on PATH, then: pip install cairosvg",
            )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / "spm"

    dataset_names = [args.dataset] + list(args.extra_datasets or [])
    row_limits: dict[str, int] = {}
    if args.fonts_subsample is not None:
        row_limits["starvector/svg-fonts-simple"] = args.fonts_subsample
    print(f"Loading and merging: {dataset_names}")
    if row_limits:
        print(f"Row limits: {row_limits}")
    rows = _merge_hf_datasets(dataset_names, row_limits=row_limits or None)
    print(f"Unique files (merged): {len(rows)}")

    cleaned, clean_stats = _clean_rows(
        rows,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        render_check=args.render_check,
    )
    print(
        "Cleaning stats:",
        dict(clean_stats) if clean_stats else "no drops",
        f"kept={len(cleaned)}",
    )

    train, val, test = _split_rows(
        cleaned,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(f"Split sizes (files): train={len(train)} val={len(val)} test={len(test)}")

    print("Training SentencePiece BPE on train split ...")
    _train_sentencepiece(train, prefix_path=model_prefix, vocab_size=args.vocab_size)
    sp_model = model_prefix.with_suffix(".model")
    processor = spm.SentencePieceProcessor(model_file=str(sp_model))

    train_f, drop_tr = _filter_by_tokens(train, processor, args.max_token_len)
    val_f, drop_va = _filter_by_tokens(val, processor, args.max_token_len)
    test_f, drop_te = _filter_by_tokens(test, processor, args.max_token_len)

    train_tokens = _total_tokens(train_f)
    val_tokens = _total_tokens(val_f)
    test_tokens = _total_tokens(test_f)

    print(
        "Token totals:",
        f"train={train_tokens}",
        f"val={val_tokens}",
        f"test={test_tokens}",
    )
    print(
        "Dropped by token cap:",
        f"train={drop_tr}",
        f"val={drop_va}",
        f"test={drop_te}",
    )

    if train_tokens < args.min_train_tokens:
        print(
            f"WARNING: train tokens {train_tokens} < recommended minimum {args.min_train_tokens}. "
            "Add svg-emoji-simple or subsample svg-fonts-simple (see project PDF).",
        )

    _write_jsonl(out_dir / "train.jsonl", train_f, keep_meta=args.keep_meta)
    _write_jsonl(out_dir / "val.jsonl", val_f, keep_meta=args.keep_meta)
    _write_jsonl(out_dir / "test.jsonl", test_f, keep_meta=args.keep_meta)

    train_lengths = [int(r["num_tokens"]) for r in train_f]

    report = {
        "datasets": dataset_names,
        "seed": args.seed,
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "cleaning": dict(clean_stats),
        "kept_files_after_clean": len(cleaned),
        "files_after_token_filter": {
            "train": len(train_f),
            "val": len(val_f),
            "test": len(test_f),
        },
        "dropped_by_token_cap": {
            "train": drop_tr,
            "val": drop_va,
            "test": drop_te,
        },
        "tokens": {
            "train": train_tokens,
            "val": val_tokens,
            "test": test_tokens,
        },
        "tokenizer": {
            "kind": "sentencepiece_bpe",
            "vocab_size": args.vocab_size,
            "model_path": sp_model.as_posix(),
            "max_token_len": args.max_token_len,
        },
        "train_length_histogram": _length_histogram(train_lengths),
    }

    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {stats_path}")
    print(f"Wrote jsonl + tokenizer under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
