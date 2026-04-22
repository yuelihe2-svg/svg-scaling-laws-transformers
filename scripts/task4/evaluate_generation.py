"""
Part 4: Quantitative evaluation.

Computes:
  - test perplexity (approx, using random batches like val)
  - XML validity rate (lxml.etree parsing)
  - structural validity rate (simple SVG root checks)
  - render rate (CairoSVG -> PNG) if enabled / pngs exist

Example:
  python -m scripts.task4.evaluate_generation \
    --test-jsonl data/processed/test.jsonl --spm-model data/processed/spm.model \
    --checkpoint outputs/task4/best_xl_sp/checkpoints/final.pt --config outputs/task4/best_xl_sp/config.json \
    --samples-dir outputs/task4/samples --out-json outputs/task4/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.task2.data import estimate_val_loss, load_jsonl_token_stream
from scripts.task2.model import SVGTransformerLM


def _xml_valid(svg_text: str) -> bool:
    try:
        from lxml import etree

        etree.fromstring(svg_text.encode("utf-8"))
        return True
    except Exception:
        return False


def _struct_valid(svg_text: str) -> bool:
    try:
        from lxml import etree

        root = etree.fromstring(svg_text.encode("utf-8"))
        # handle namespaces like {http://www.w3.org/2000/svg}svg
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag.lower() != "svg":
            return False
        # basic attribute sanity: at least one of viewBox / width/height present
        has_vb = "viewBox" in root.attrib or "viewbox" in {k.lower(): v for k, v in root.attrib.items()}
        has_wh = ("width" in root.attrib) or ("height" in root.attrib)
        return bool(has_vb or has_wh)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-jsonl", type=Path, required=True)
    ap.add_argument("--spm-model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--samples-dir", type=Path, required=True, help="Directory produced by sample_generate.py")
    ap.add_argument("--out-json", type=Path, default=Path("outputs/task4/eval_metrics.json"))
    ap.add_argument("--test-batches", type=int, default=120, help="More batches -> stabler perplexity estimate.")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    block_size = int(cfg["block_size"])
    tokens_per_batch = int(cfg["tokens_per_batch"])
    batch_size = int(tokens_per_batch // block_size)
    vocab_size = int(cfg["vocab_size"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # model
    model = SVGTransformerLM(
        vocab_size=vocab_size,
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        d_ff=int(cfg["d_ff"]),
        max_seq_len=block_size,
        dropout=0.0,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # test perplexity (approx via random batches on test stream)
    test_data, vs = load_jsonl_token_stream(args.test_jsonl, args.spm_model)
    if vs != vocab_size:
        raise RuntimeError(f"Vocab mismatch: test={vs} config={vocab_size}")
    rng = np.random.default_rng(args.seed)
    test_nll = float(estimate_val_loss(model, test_data, batch_size, block_size, device, rng, args.test_batches))
    ppl = float(np.exp(test_nll))

    # generated samples validity
    manifest_path = args.samples_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    xml_ok = 0
    struct_ok = 0
    render_ok = 0
    n = 0
    for rec in manifest:
        svg_path = Path(rec["svg_path"])
        txt = svg_path.read_text(encoding="utf-8", errors="ignore")
        n += 1
        xo = _xml_valid(txt)
        so = _struct_valid(txt)
        xml_ok += int(xo)
        struct_ok += int(so)
        # render: prefer existing rec flag; if missing, count only if png exists and non-empty
        if "render_ok" in rec:
            render_ok += int(bool(rec["render_ok"]))
        else:
            png_path = rec.get("png_path")
            if png_path and Path(png_path).exists() and Path(png_path).stat().st_size > 0:
                render_ok += 1

    metrics = {
        "test_nll": test_nll,
        "test_perplexity": ppl,
        "num_generated": n,
        "xml_valid_rate": float(xml_ok / max(1, n)),
        "struct_valid_rate": float(struct_ok / max(1, n)),
        "render_rate": float(render_ok / max(1, n)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Wrote", args.out_json)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

