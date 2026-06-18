"""Tokenizer-backed PyTorch Dataset + DataLoader factory.

Reads a CSV manifest with at minimum (text, label) columns, tokenizes with the
RoBERTa tokenizer on the fly, and pads dynamically inside the collate fn.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


class TextEmotionDataset(Dataset):
    """Holds raw (text, label) rows; tokenization is lazy in __getitem__."""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.df = pd.read_csv(manifest_path)
        missing = {text_column, label_column} - set(self.df.columns)
        if missing:
            raise KeyError(f"{manifest_path}: missing columns {missing}")
        self.texts: List[str] = self.df[text_column].astype(str).tolist()
        self.labels: List[int] = self.df[label_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def _collate_fn(pad_token_id: int):
    def _inner(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.empty(len(batch), dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["input_ids"].size(0)
            input_ids[i, :n] = item["input_ids"]
            attention_mask[i, :n] = item["attention_mask"]
            labels[i] = item["label"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return _inner


def build_dataloaders(
    manifest_dir: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    train_batch_size: int,
    eval_batch_size: int,
    max_length: int,
    num_workers: int = 4,
    splits: Tuple[str, ...] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    """Return a {split: DataLoader} dict. Splits whose CSV is missing are skipped."""
    manifest_dir = Path(manifest_dir)
    collate = _collate_fn(tokenizer.pad_token_id)
    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        csv = manifest_dir / f"{split}.csv"
        if not csv.exists():
            continue
        ds = TextEmotionDataset(csv, tokenizer=tokenizer, max_length=max_length)
        loaders[split] = DataLoader(
            ds,
            batch_size=train_batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate,
            drop_last=(split == "train"),
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders


def compute_class_weights(manifest_path: str | Path, num_labels: int) -> torch.Tensor:
    """Inverse-frequency weights for CE loss. Returns a tensor of shape (num_labels,)."""
    df = pd.read_csv(manifest_path)
    counts = df["label"].value_counts().reindex(range(num_labels)).fillna(0).to_numpy()
    counts = counts.clip(min=1.0)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float32)
