"""Decoder-only Transformer in μP style (MuReadout, 1/d attention, separate Q/K/V)."""

from __future__ import annotations

import mup.init as mup_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout, set_base_shapes


class CausalSelfAttentionMuP(nn.Module):
    """Causal attention with **1 / d_head** scaling (μP / assignment), not 1/sqrt(d_head)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, dh = self.n_heads, self.d_head
        q = self.wq(x).view(B, T, nh, dh).transpose(1, 2)
        k = self.wk(x).view(B, T, nh, dh).transpose(1, 2)
        v = self.wv(x).view(B, T, nh, dh).transpose(1, 2)
        # μP / course: scale logits by 1/d_head (not 1/sqrt(d_head))
        att = (q @ k.transpose(-2, -1)) * (1.0 / float(dh))
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionMuP(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SVGTransformerLM_MuP(nn.Module):
    """Causal LM: μP readout + 1/d attention; call `apply_mup_base_shapes` after init."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = MuReadout(d_model, vocab_size, bias=False, readout_zero_init=True)

    def init_weights_post_mup(self) -> None:
        """Call **after** `set_base_shapes`. Uses `mup.init` for fan-aware scaling."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                mup_init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                mup_init.normal_(m.weight, mean=0.0, std=0.02)
        # μP / mup README: zero-init query projection for stable attention logits at init
        for block in self.blocks:
            assert isinstance(block.attn, CausalSelfAttentionMuP)
            nn.init.zeros_(block.attn.wq.weight)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_seq_len
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def build_model_for_preset(
    preset,
    *,
    vocab_size: int,
    max_seq_len: int,
    dropout: float,
) -> SVGTransformerLM_MuP:
    return SVGTransformerLM_MuP(
        vocab_size=vocab_size,
        d_model=preset.d_model,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        d_ff=preset.d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )


def apply_mup_base_shapes(
    model: SVGTransformerLM_MuP,
    *,
    base_preset,
    delta_preset,
    dropout: float = 0.0,
) -> SVGTransformerLM_MuP:
    """Attach `infshape` and rescale init for μP. `base_preset` / `delta_preset` must match depth/heads layout."""
    vs = int(model.head.out_features)
    T = int(model.pos_emb.num_embeddings)
    base = build_model_for_preset(base_preset, vocab_size=vs, max_seq_len=T, dropout=dropout)
    delta = build_model_for_preset(delta_preset, vocab_size=vs, max_seq_len=T, dropout=dropout)
    return set_base_shapes(model, base, delta=delta)
