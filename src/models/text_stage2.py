"""Stage II text model: Bi-GRU + multi-head self-attention over utterance features.

Architecture (matches Fig.1 of the paper, "Bi-GRU with self-attention" block):

    (K, D_in) utterance features  -- D_in = 1024 from Stage I RoBERTa-FT
            ↓
    Bi-GRU (hidden = gru_hidden, 1 layer)         -> (K, 2*gru_hidden)
            ↓
    LayerNorm
            ↓
    Multi-head self-attention with padding mask    -> (K, 2*gru_hidden)
    + residual + LayerNorm
            ↓
    FC (2*gru_hidden -> num_labels)               -> (K, num_labels)

The encoder (RoBERTa-FT + dense) is NOT inside this module — its features
are pre-computed by scripts/extract_text_features.py and read from disk by
DialogueDataset. That matches the paper's "keep previous stage frozen"
training recipe (Sec. III-C).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextStage2(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        gru_hidden: int = 256,
        num_heads: int = 4,
        num_labels: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        feature_dim = 2 * gru_hidden

        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads}). "
                f"Try num_heads in {{1, 2, 4, 8}}."
            )

        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln_gru = nn.LayerNorm(feature_dim)
        self.ln_attn = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_labels)

        self.num_labels = num_labels
        self.feature_dim = feature_dim

    def encode(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the contextualized utterance embedding T²_k for Stage III."""
        gru_out, _ = self.bigru(features)                # (B, K, 2H)
        gru_out = self.ln_gru(gru_out)

        # MultiheadAttention: key_padding_mask is True where positions are padded.
        key_padding_mask = ~mask                          # (B, K)  True = ignore
        attn_out, _ = self.attn(
            gru_out, gru_out, gru_out,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.ln_attn(gru_out + self.dropout(attn_out))   # (B, K, 2H)
        return x

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        features: (B, K, D_in)
        mask:     (B, K) bool, True = real utterance, False = pad
        labels:   (B, K) long, -100 for pad slots
        """
        x = self.encode(features, mask)                  # (B, K, 2H)
        logits = self.classifier(x)                      # (B, K, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits, "features": x}


def build_text_stage2(cfg) -> TextStage2:
    return TextStage2(
        input_dim=int(cfg.input_dim),
        gru_hidden=int(cfg.gru_hidden),
        num_heads=int(cfg.num_heads),
        num_labels=int(cfg.num_labels),
        dropout=float(cfg.dropout),
    )
