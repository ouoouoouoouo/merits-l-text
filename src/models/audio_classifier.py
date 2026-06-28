"""Audio Stage I classifier — FC head on precomputed WavLM (or similar) features.

Supports two feature shapes:
  - (B, D)               : single-layer features (e.g., last hidden state mean-pooled).
                            No layer weighting; just FC.
  - (B, L, D)            : per-layer features (1 + N transformer layers), SUPERB-style.
                            Applies learnable softmax-normalized convex combination over
                            the L axis before the FC head (matches CARE's inference
                            paradigm in Sec III-A-3 of the MERITS-L paper).

Architecture (Fig.1 of the paper, "CARE → FC" block):

    feature (D,) or (L, D)
        ↓ [convex combination if L>1] : sum(softmax(w_i) * h_i)
    Linear(D, D) → tanh → dropout → Linear(D, num_labels)
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
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_layers = int(num_layers) if num_layers and num_layers > 1 else 0
        if self.num_layers > 0:
            # Init at zeros so softmax starts uniform across layers.
            self.layer_weights = nn.Parameter(torch.zeros(self.num_layers))

        self.dense = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, num_labels)
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def _combine_layers(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, L, D) → (B, D)  via softmax-weighted sum across L."""
        if features.dim() != 3:
            return features
        if self.num_layers == 0:
            raise ValueError(
                f"Got {features.size(1)}-layer features but model.num_layers is 0. "
                f"Set `num_layers: {features.size(1)}` in the config."
            )
        if features.size(1) != self.num_layers:
            raise ValueError(
                f"feature layer count {features.size(1)} != model.num_layers {self.num_layers}"
            )
        w = F.softmax(self.layer_weights, dim=0)         # (L,)
        return (features * w[None, :, None]).sum(dim=1)  # (B, D)

    def get_features(self, features: torch.Tensor) -> torch.Tensor:
        """Hidden representation (B, hidden_dim) used as S¹_k for Stage II."""
        x = self._combine_layers(features)
        x = self.dropout(x)
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

    @torch.no_grad()
    def get_layer_weights(self) -> torch.Tensor:
        """Diagnostic: return current softmax-normalized layer weights."""
        if self.num_layers == 0:
            return torch.tensor([])
        return F.softmax(self.layer_weights, dim=0).detach().cpu()


def build_audio_classifier(cfg) -> AudioClassifier:
    return AudioClassifier(
        input_dim=int(cfg.input_dim),
        hidden_dim=int(cfg.get("hidden_dim", cfg.input_dim)) if hasattr(cfg, "get") else int(cfg.input_dim),
        num_labels=int(cfg.num_labels),
        dropout=float(cfg.dropout),
        num_layers=cfg.get("num_layers", None) if hasattr(cfg, "get") else None,
    )
