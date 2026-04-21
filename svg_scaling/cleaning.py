"""Normalize and validate SVG text for language-model training."""

from __future__ import annotations

import os
import re
from typing import Any, Final

from lxml import etree

_FLOAT_RE: Final[str] = r"-?\d+\.\d+(?:[eE][+-]?\d+)?"
_STRIP_TAGS: Final[frozenset[str]] = frozenset({"metadata", "title", "desc"})


def _round_floats(text: str, ndigits: int = 1) -> str:
    """Round decimal literals to reduce vocabulary size (assignment suggestion)."""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            val = float(raw)
        except ValueError:
            return raw
        rounded = round(val, ndigits)
        if ndigits <= 0:
            return str(int(rounded))
        s = f"{rounded:.{ndigits}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s else "0"

    return re.sub(_FLOAT_RE, repl, text)


def _remove_elements(root: Any, localnames: frozenset[str]) -> None:
    to_drop: list[Any] = []
    for el in root.iter():
        if etree.QName(el).localname in localnames:
            to_drop.append(el)
    for el in to_drop:
        parent = el.getparent()
        if parent is not None:
            parent.remove(el)


def clean_svg(raw: str) -> str | None:
    """
    Strip comments and soft metadata, normalize whitespace, round floats, return XML string.
    Returns None if the input is not well-formed enough for lxml to parse as XML.
    """
    parser = etree.XMLParser(remove_comments=True, recover=True, huge_tree=True)
    try:
        root = etree.fromstring(raw.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return None

    _remove_elements(root, _STRIP_TAGS)

    try:
        xml_bytes = etree.tostring(root, encoding="utf-8", xml_declaration=False)
    except etree.XMLSyntaxError:
        return None

    text = xml_bytes.decode("utf-8")
    text = re.sub(r">\s+<", "><", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _round_floats(text, ndigits=1)

    # Re-parse after numeric edits to ensure still valid XML
    try:
        etree.fromstring(text.encode("utf-8"), etree.XMLParser(recover=False))
    except etree.XMLSyntaxError:
        return None

    return text


def _register_cairo_dll_dirs() -> None:
    """Windows: let the loader see GTK/Cairo DLLs before cairocffi imports (Python 3.8+)."""
    if os.name != "nt":
        return
    add = getattr(os, "add_dll_directory", None)
    if not callable(add):
        return
    seen: set[str] = set()
    parts: list[str] = []
    env = os.environ.get("CAIROCFFI_DLL_DIRECTORIES", "").strip()
    if env:
        parts.extend(p.strip() for p in env.split(os.pathsep) if p.strip())
    # Typical GTK3 Runtime installs (drive letter varies)
    for p in (
        r"C:\Program Files\GTK3-Runtime Win64\bin",
        r"D:\Program Files\GTK3-Runtime Win64\bin",
    ):
        if p not in parts:
            parts.append(p)
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        cairo = os.path.join(p, "libcairo-2.dll")
        if not os.path.isfile(cairo):
            continue
        # libcairo depends on zlib/fontconfig/etc.; they live in the same bin — prepend PATH.
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        try:
            add(p)
        except OSError:
            continue


def cairosvg_available() -> tuple[bool, str]:
    """Whether CairoSVG can import and load native Cairo (needed on Windows for DLLs)."""
    _register_cairo_dll_dirs()
    try:
        import cairosvg  # noqa: F401
    except ImportError as e:
        return False, f"import failed: {e}"
    except OSError as e:
        return False, f"native Cairo missing or broken (common on Windows): {e}"
    return True, "ok"


def try_render_svg(svg: str) -> bool:
    """Return True if CairoSVG can rasterize the SVG (optional quality gate)."""
    ok, _ = cairosvg_available()
    if not ok:
        return False
    import cairosvg

    try:
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=256,
            output_height=256,
        )
        return True
    except Exception:
        return False
