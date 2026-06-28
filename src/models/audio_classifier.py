"""Audio Stage I classifier — FC head on precomputed WavLM (or similar) features.

Architecture (matches Fig.1 of the paper, "CARE → FC" block):

    feature (D,)  →  Linear(D, D) → tanh → dropout → Linear(D, num_labels)

The encoder (WavLM-base) is NOT inside this module — its features are pre-
computed by scripts/extract_audio_features.py and cached as .pt. Matches the
paper's "frozen CARE embeddings as input" setup (Sec III-C).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_labels: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

    def get_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return the hidden representation (B, hidden_dim) used as S¹_k for Stage II."""
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return x

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        h = self.get_features(features)
        logits = self.out_proj(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "features": h}


def build_audio_classifier(cfg) -> AudioClassifier:
    return AudioClassifier(
        input_dim=int(cfg.input_dim),
        hidden_dim=int(cfg.get("hidden_dim", cfg.input_dim)) if hasattr(cfg, "get") else int(cfg.input_dim),
        num_labels=int(cfg.num_labels),
        dropout=float(cfg.dropout),
    )
