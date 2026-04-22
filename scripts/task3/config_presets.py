"""
μP ladder presets: **same depth (12) and same n_heads (8)** for every size so
`mup.set_base_shapes(model, base, delta)` can match module trees across widths.

This intentionally **does not** match Part 2 layer counts for tiny/small/medium/large
(Part 2 uses 4/6/6/10/12). **XL** matches Part 2 parameter count exactly when
vocab_size matches. Document this in the report when comparing SP vs μP curves.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MuModelPreset:
    name: str
    approx_params: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


MU_PRESETS: dict[str, MuModelPreset] = {
    "tiny": MuModelPreset("tiny", "~3M μP", 128, 12, 8, 512),
    "small": MuModelPreset("small", "~7M μP", 192, 12, 8, 768),
    "medium": MuModelPreset("medium", "~24M μP", 384, 12, 8, 1536),
    "large": MuModelPreset("large", "~42M μP", 512, 12, 8, 2048),
    "xl": MuModelPreset("xl", "~88M μP", 768, 12, 8, 3072),
}
