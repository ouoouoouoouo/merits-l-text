"""MSP-PODCAST pseudo-label manifest helpers.

The pretrain stage trains RoBERTa-large on (transcript, pseudo_label) pairs
where the pseudo labels come from a text LLM (GPT-3.5 Turbo in the paper). This
module just splits an existing labelled CSV into train/val 80/20.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_manifest(
    manifest_path: str | Path,
    label_map: Dict[str, int],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{manifest_path}: missing columns {missing}. Expecting at least "
            "(utt_id,text,label) — produce it with scripts/llm_pseudo_label_msp.py."
        )
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label_str"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label_str"].isin(label_map)].reset_index(drop=True)
    df["label"] = df["label_str"].map(label_map).astype(int)
    if "utt_id" not in df.columns:
        df["utt_id"] = df.index.astype(str)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = int(round(len(df) * val_fraction))
    val_idx = set(idx[:n_val].tolist())
    df["split"] = ["val" if i in val_idx else "train" for i in range(len(df))]

    out_cols = ["utt_id", "text", "label", "split"]
    return (
        df[df.split == "train"][out_cols].reset_index(drop=True),
        df[df.split == "val"][out_cols].reset_index(drop=True),
    )


def write_manifests(
    manifest_path: str | Path,
    manifest_dir: str | Path,
    label_map: Dict[str, int],
    val_fraction: float,
    seed: int,
) -> Dict[str, Path]:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df = build_manifest(manifest_path, label_map, val_fraction, seed)
    paths = {}
    for split, df in [("train", train_df), ("val", val_df)]:
        out = manifest_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        paths[split] = out
    return paths
