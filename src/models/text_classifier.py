"""RoBERTa-large + FC head — Stage I text classifier.

Architecture (matches Fig. 1 of the paper, top branch):
    Transcript -> RoBERTa-large -> pooled embedding -> FC -> emotion logits

The `pretrained` argument can be either a HuggingFace name (e.g. `roberta-large`)
or a local directory holding a checkpoint saved by `save_pretrained`. The
downstream Stage I uses the MSP-PODCAST pre-trained "RoBERTa-FT" as init when
available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class RobertaTextClassifier(nn.Module):
    """RoBERTa encoder + (dropout + linear) classification head.

    Pooling rule follows HuggingFace's `RobertaClassificationHead`: take the
    final hidden state of the <s> (CLS) token, apply tanh, dropout, then a
    linear layer to `num_labels`.
    """

    def __init__(
        self,
        pretrained: str | Path = "roberta-large",
        num_labels: int = 4,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(pretrained)
        self.encoder = AutoModel.from_pretrained(pretrained, config=config)
        hidden_size = config.hidden_size

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.num_labels = num_labels
        self.config = config

    def get_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pooled (B, H) utterance embedding — used downstream by Stage II."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # <s> token
        x = self.dropout(cls)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        features = self.get_features(input_ids, attention_mask)
        logits = self.out_proj(features)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits, "features": features}


def build_model(cfg) -> RobertaTextClassifier:
    """Instantiate the model from a config dataclass / dict-like."""
    return RobertaTextClassifier(
        pretrained=cfg.pretrained,
        num_labels=cfg.num_labels,
        dropout=cfg.dropout,
        freeze_encoder=cfg.freeze_encoder,
    )


def build_tokenizer(pretrained: str | Path):
    """Wrap AutoTokenizer with `use_fast=True` (default) and add no special tokens.

    Returning a function so the trainer can pass the same identifier for model
    and tokenizer — useful when loading a local RoBERTa-FT checkpoint that
    includes its own tokenizer files.
    """
    return AutoTokenizer.from_pretrained(pretrained, use_fast=True)
