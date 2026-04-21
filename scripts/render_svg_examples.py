"""
Export PNG and/or SVG examples at different BPE lengths for the PDF report.

PNG order (first success wins):
  1) CairoSVG (needs native Cairo DLL — often missing on Windows)
  2) Inkscape CLI (if `inkscape` on PATH)
  3) ImageMagick v7 (if `magick` on PATH)

gallery.html embeds each SVG **inline** (raw <svg>...</svg>) so file:// works in Edge/Chrome.
Use a real browser — not Notepad, not the Cursor/VS Code preview.

Usage (from repo root):
  python scripts/render_svg_examples.py --jsonl data/processed/train.jsonl --out-dir outputs/figures
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_name(s: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", s)[:120]


def _svg_for_inline_html(svg: str) -> str:
    """Strip XML declaration so inline SVG parses reliably in HTML5."""
    s = svg.strip()
    if s.startswith("<?xml"):
        end = s.find("?>")
        if end != -1:
            s = s[end + 2 :].lstrip()
    return s


def _cairo_png(svg: str, png_path: Path, size: int) -> bool:
    try:
        import cairosvg
    except (ImportError, OSError):
        return False
    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(png_path),
            output_width=size,
            output_height=size,
        )
        return png_path.exists() and png_path.stat().st_size > 0
    except OSError:
        return False


def _inkscape_png(svg_path: Path, png_path: Path, size: int) -> bool:
    exe = shutil.which("inkscape")
    if not exe:
        return False
    # Inkscape 1.x
    r = subprocess.run(
        [exe, str(svg_path), "-o", str(png_path), "-w", str(size), "-h", str(size)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0 and png_path.exists() and png_path.stat().st_size > 0


def _magick_png(svg_path: Path, png_path: Path, size: int) -> bool:
    exe = shutil.which("magick")
    if not exe:
        return False
    r = subprocess.run(
        [exe, str(svg_path), "-resize", f"{size}x{size}", str(png_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return r.returncode == 0 and png_path.exists() and png_path.stat().st_size > 0


def _try_all_png(
    svg: str,
    svg_path: Path,
    png_path: Path,
    size: int,
    *,
    skip_cairo: bool,
) -> str | None:
    if not skip_cairo and _cairo_png(svg, png_path, size):
        return "cairosvg"
    if _inkscape_png(svg_path, png_path, size):
        return "inkscape"
    if _magick_png(svg_path, png_path, size):
        return "imagemagick"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True, help="Usually data/processed/train.jsonl")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "figures")
    p.add_argument("--png-size", type=int, default=256, help="Raster width/height in pixels.")
    p.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 0.9],
        help="Which quantiles of num_tokens to sample (0–1).",
    )
    p.add_argument(
        "--png-only-tools",
        action="store_true",
        help="Skip CairoSVG; try Inkscape/ImageMagick only (useful if Cairo DLL is broken).",
    )
    args = p.parse_args()

    rows: list[dict] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No rows in jsonl.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: int(r["num_tokens"]))
    n = len(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Picked {len(args.quantiles)} examples from {n} rows (by num_tokens).")
    html_sections: list[str] = []
    png_methods: list[str] = []

    for q in args.quantiles:
        q = min(1.0, max(0.0, q))
        idx = int(round(q * (n - 1)))
        r = rows[idx]
        svg = r["svg"]
        nt = r["num_tokens"]
        ds = r.get("source_dataset", "unknown")
        fid = r.get("filename", idx)
        base = _safe_name(f"q{q:.2f}_tok{nt}_ds{ds}_id{fid}")
        svg_name = f"{base}.svg"
        svg_path = args.out_dir / svg_name
        png_path = args.out_dir / f"{base}.png"
        svg_path.write_text(svg, encoding="utf-8")

        method = _try_all_png(
            svg,
            svg_path,
            png_path,
            args.png_size,
            skip_cairo=args.png_only_tools,
        )

        if method:
            png_methods.append(method)
            print(f"  {png_path}  (num_tokens={nt}, quantile≈{q}, via {method})")
        else:
            print(f"  {svg_path}  (num_tokens={nt}, quantile≈{q}, PNG skipped — see gallery.html)")

        cap = html.escape(f"Quantile≈{q:.2f}, num_tokens={nt}, source={ds}, filename={fid}")
        svg_inline = _svg_for_inline_html(svg)
        html_sections.append(
            "<section style='margin-bottom:2em;border:1px solid #ccc;padding:1em;'>"
            f"<p><strong>{cap}</strong></p>"
            f'<div style="width:{args.png_size}px;height:{args.png_size}px;'
            f"border:1px solid #ddd;background:#f8f8f8;overflow:auto;\">"
            f"{svg_inline}"
            f"</div>"
            "</section>"
        )

    gallery = args.out_dir / "gallery.html"
    gallery.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'><title>SVG examples</title></head>"
        "<body style='font-family:sans-serif;max-width:900px;margin:1em auto;'>"
        "<h1>SVG complexity examples (by BPE length)</h1>"
        "<p>Figures are <strong>inline SVG</strong> below (works with file://). "
        "If you see nothing: use <strong>Edge or Chrome</strong> (not Notepad / not Cursor preview). "
        "Right-click the file → Open with → Microsoft Edge.</p>"
        + "\n".join(html_sections)
        + "</body></html>",
        encoding="utf-8",
    )

    readme = args.out_dir / "README.txt"
    readme.write_text(
        "gallery.html — open in Edge/Chrome; images are embedded (base64), no local file block.\n"
        ".svg / .png — same examples; PNG needs Cairo, Inkscape, or ImageMagick.\n",
        encoding="utf-8",
    )
    print(f"Wrote {gallery}")
    print(
        "Tip: open gallery.html with Edge or Chrome (double-click may use the wrong app).",
    )
    if png_methods:
        print(f"PNG backends used: {sorted(set(png_methods))}")
    else:
        print(
            "No PNG written. Install one of: Cairo (for cairosvg), "
            "Inkscape (https://inkscape.org), or ImageMagick (https://imagemagick.org), "
            "then re-run this script.",
            file=sys.stderr,
        )
    print(f"Outputs under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
