"""
Part 4: Generate unconditional samples and prefix-conditioned completions.

Writes SVG code + (optionally) rendered PNGs to an output directory.

Example:
  python -m scripts.task4.sample_generate \
    --spm-model data/processed/spm.model \
    --checkpoint outputs/task4/best_xl_sp/checkpoints/final.pt \
    --config outputs/task4/best_xl_sp/config.json \
    --out-dir outputs/task4/samples \
    --num-uncond 10 --num-prefix 5 \
    --temperatures 0.5 0.8 1.0 \
    --top-k 50 --top-p 0.95
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from scripts.task2.model import SVGTransformerLM


def _load_sp(spm_model: Path):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(spm_model))


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> int:
    # logits: (vocab,)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)

    if top_k is not None and top_k > 0:
        v, idx = torch.topk(probs, k=min(top_k, probs.numel()))
        v = v / v.sum()
        choice = int(idx[torch.multinomial(v, 1)].item())
        return choice

    if top_p is not None and 0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=0)
        mask = cdf <= float(top_p)
        # ensure at least one token
        if not bool(mask.any()):
            mask[0] = True
        keep_idx = sorted_idx[mask]
        keep_probs = sorted_probs[mask]
        keep_probs = keep_probs / keep_probs.sum()
        choice = int(keep_idx[torch.multinomial(keep_probs, 1)].item())
        return choice

    # full multinomial
    return int(torch.multinomial(probs, 1).item())


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    prefix: list[int],
    *,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    eos_id: int | None,
    device: torch.device,
) -> list[int]:
    ids = list(prefix)
    for _ in range(max_new_tokens):
        ctx = ids[-block_size:]
        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)[0, -1]  # (vocab,)
        nxt = _sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        ids.append(nxt)
        if eos_id is not None and nxt == eos_id:
            break
    return ids


def _default_prefixes() -> list[str]:
    # Keep these minimal; they are meant to be *partial* SVGs.
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="24" cy="28" r="6"/>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M10 32 L54 32',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><g><rect x="8" y="8" width="16" height="16"/></g>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="20"/><circle cx="24" cy="28" r="3"/>',
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><path d="M16 16 C 20 10, 44 10, 48 16',
    ]


def _render_png(svg_text: str, out_png: Path) -> bool:
    try:
        import cairosvg

        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(out_png))
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spm-model", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="config.json produced by training")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/task4/samples"))
    ap.add_argument("--num-uncond", type=int, default=10)
    ap.add_argument("--num-prefix", type=int, default=5)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render", action="store_true", help="If set, try to render to PNG with CairoSVG.")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    block_size = int(cfg["block_size"])
    vocab_size = int(cfg["vocab_size"])
    # model shape
    d_model = int(cfg["d_model"])
    n_layers = int(cfg["n_layers"])
    n_heads = int(cfg["n_heads"])
    d_ff = int(cfg["d_ff"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    sp = _load_sp(args.spm_model)
    eos_id = sp.eos_id()
    if eos_id < 0:
        eos_id = None

    model = SVGTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=block_size,
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "svg").mkdir(parents=True, exist_ok=True)
    if args.render:
        (out / "png").mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    # Unconditional: always start from "<svg"
    uncond_prefix = "<svg"
    uncond_ids = sp.encode(uncond_prefix, out_type=int)

    temps = list(args.temperatures)
    # Generate num_uncond across temperatures (round-robin)
    for i in range(args.num_uncond):
        t = temps[i % len(temps)]
        ids = generate_tokens(
            model,
            uncond_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            temperature=t,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            eos_id=eos_id,
            device=device,
        )
        txt = sp.decode(ids)
        svg_path = out / "svg" / f"uncond_{i:02d}_t{t:g}.svg"
        svg_path.write_text(txt, encoding="utf-8")
        rec = {"type": "unconditional", "i": i, "temperature": t, "top_k": args.top_k, "top_p": args.top_p, "svg_path": str(svg_path)}
        if args.render:
            png_path = out / "png" / f"uncond_{i:02d}_t{t:g}.png"
            rec["render_ok"] = _render_png(txt, png_path)
            rec["png_path"] = str(png_path)
        manifest.append(rec)

    # Prefix-conditioned
    prefixes = _default_prefixes()[: max(1, args.num_prefix)]
    for i in range(args.num_prefix):
        t = temps[i % len(temps)]
        prefix = prefixes[i % len(prefixes)]
        prefix_ids = sp.encode(prefix, out_type=int)
        ids = generate_tokens(
            model,
            prefix_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=block_size,
            temperature=t,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            eos_id=eos_id,
            device=device,
        )
        txt = sp.decode(ids)
        svg_path = out / "svg" / f"prefix_{i:02d}_t{t:g}.svg"
        svg_path.write_text(txt, encoding="utf-8")
        rec = {
            "type": "prefix",
            "i": i,
            "temperature": t,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "prefix": prefix,
            "svg_path": str(svg_path),
        }
        if args.render:
            png_path = out / "png" / f"prefix_{i:02d}_t{t:g}.png"
            rec["render_ok"] = _render_png(txt, png_path)
            rec["png_path"] = str(png_path)
        manifest.append(rec)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

