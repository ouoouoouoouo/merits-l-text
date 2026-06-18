"""MELD text manifest builder.

The official MELD release ships three CSVs under `MELD.Raw/` (or `MELD/`):
    train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv

Each row has columns including `Utterance`, `Emotion`, `Dialogue_ID`,
`Utterance_ID`, `Speaker`, `Sentiment`. Paper uses 7-way emotion labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def _load_split(csv_path: Path, text_col: str, label_col: str, label_map: Dict[str, int]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path}: missing columns {missing}; got {list(df.columns)}")
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()
    df = df[df[label_col].isin(label_map)].reset_index(drop=True)
    df["text"] = df[text_col]
    df["raw_emotion"] = df[label_col]
    df["label"] = df["raw_emotion"].map(label_map).astype(int)
    # Build a stable utt_id if not present.
    if "utt_id" not in df.columns:
        if {"Dialogue_ID", "Utterance_ID"}.issubset(df.columns):
            df["utt_id"] = "dia" + df["Dialogue_ID"].astype(str) + "_utt" + df["Utterance_ID"].astype(str)
        else:
            df["utt_id"] = df.index.astype(str)
    df["dialogue_id"] = df.get("Dialogue_ID", "").astype(str)
    return df[["utt_id", "dialogue_id", "text", "raw_emotion", "label"]]


def build_manifest(
    root: str | Path,
    csv_train: str,
    csv_val: str,
    csv_test: str,
    text_column: str,
    label_column: str,
    label_map: Dict[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    lm = {k.lower(): v for k, v in label_map.items()}

    def _find(name: str) -> Path:
        # Tolerate both MELD/ and MELD.Raw/ layouts.
        for cand in (root / name, root / "MELD.Raw" / name, root / "MELD" / name):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"MELD CSV not found: {name} under {root}")

    train_df = _load_split(_find(csv_train), text_column, label_column, lm).assign(split="train")
    val_df = _load_split(_find(csv_val), text_column, label_column, lm).assign(split="val")
    test_df = _load_split(_find(csv_test), text_column, label_column, lm).assign(split="test")
    return train_df, val_df, test_df


def write_manifests(
    root: str | Path,
    manifest_dir: str | Path,
    csv_train: str,
    csv_val: str,
    csv_test: str,
    text_column: str,
    label_column: str,
    label_map: Dict[str, int],
) -> Dict[str, Path]:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = build_manifest(
        root, csv_train, csv_val, csv_test, text_column, label_column, label_map
    )
    paths = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = manifest_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        paths[split] = out
    return paths
