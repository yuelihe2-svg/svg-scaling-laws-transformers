"""Suggested model shapes for Part 2 (parameter counts are approximate)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    approx_params: str
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int


PRESETS: dict[str, ModelPreset] = {
    "tiny": ModelPreset("tiny", "~1M", 128, 4, 4, 512),
    "small": ModelPreset("small", "~3M", 192, 6, 6, 768),
    "medium": ModelPreset("medium", "~10M", 384, 6, 6, 1536),
    "large": ModelPreset("large", "~30M", 512, 10, 8, 2048),
    "xl": ModelPreset("xl", "~88M", 768, 12, 12, 3072),
}
