"""Utterance-level dataset for Audio Stage I (precomputed features).

Reads (utt_id, label) from a manifest CSV and looks up the precomputed
feature from a .pt dict. Tiny, no tokenizer / no model forward — Stage I
training is just FC on cached vectors.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class AudioFeatureDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        features: Dict[str, torch.Tensor],
        utt_col: str = "utt_id",
        label_col: str = "label",
    ) -> None:
        df = pd.read_csv(manifest_path)
        for col in (utt_col, label_col):
            if col not in df.columns:
                raise KeyError(f"{manifest_path}: missing column `{col}`")
        self.features = features
        self.rows: List[Tuple[str, int]] = []
        n_missing = 0
        for _, r in df.iterrows():
            uid = str(r[utt_col])
            if uid not in features:
                n_missing += 1
                continue
            self.rows.append((uid, int(r[label_col])))
        if n_missing:
            print(f"[AudioFeatureDataset] {n_missing} rows had no feature; skipping.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, label = self.rows[idx]
        return {
            "utt_id": uid,
            "features": self.features[uid],          # (D,)
            "label": torch.tensor(label, dtype=torch.long),
        }


def _audio_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.stack([b["features"] for b in batch]).float(),  # (B, D)
        "labels": torch.stack([b["label"] for b in batch]),                # (B,)
    }


def build_audio_loaders(
    manifest_dir: str | Path,
    features_path: str | Path,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    manifest_dir = Path(manifest_dir)
    features = torch.load(str(features_path), map_location="cpu", weights_only=True)
    print(f"Loaded {len(features)} cached audio features from {features_path}")

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        csv_path = manifest_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        ds = AudioFeatureDataset(csv_path, features)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size if split == "train" else eval_batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=_audio_collate,
            drop_last=(split == "train"),
        )
    if "train" not in loaders:
        raise FileNotFoundError(f"No train.csv under {manifest_dir}")
    return loaders
