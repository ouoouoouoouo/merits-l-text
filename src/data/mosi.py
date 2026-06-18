"""CMU-MOSI text manifest builder.

The dataset comes in several flavours; the most common one for ERC reproductions
is a single CSV with columns roughly like:
    video_id, clip_id, text, sentiment (float in [-3, 3])

Paper protocol (Sec. IV-A, following Poria et al. [50]):
    - 93 monologues, 2199 utterances
    - First 62 monologues form train+val, last 31 monologues form test
    - Within the 62: 49 -> train, 13 -> val
    - Binary labels: sentiment in [-3, 0) -> negative; [0, 3] -> positive
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd


def _label_from_score(score: float) -> int:
    return 1 if score >= 0 else 0


def _normalize_video_index(video_id) -> int | None:
    """CMU-MOSI video IDs are sometimes integers, sometimes strings like
    `_dI--eQ6qVU` (YouTube IDs). When integers, they map to monologue index
    1..93 — that's what the paper's split assumes. When strings, the caller
    must provide an explicit ``train/val/test`` column instead.
    """
    try:
        return int(video_id)
    except (TypeError, ValueError):
        return None


def _split_for_monologue(
    monologue_idx: int,
    train_range: Sequence[int],
    val_range: Sequence[int],
    test_range: Sequence[int],
) -> str | None:
    if train_range[0] <= monologue_idx <= train_range[1]:
        return "train"
    if val_range[0] <= monologue_idx <= val_range[1]:
        return "val"
    if test_range[0] <= monologue_idx <= test_range[1]:
        return "test"
    return None


def build_manifest(
    csv_path: str | Path,
    text_column: str,
    score_column: str,
    id_column: str,
    train_monologues: Sequence[int],
    val_monologues: Sequence[int],
    test_monologues: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    missing = {text_column, score_column, id_column} - set(df.columns)
    if missing:
        raise KeyError(f"{csv_path}: missing columns {missing}; got {list(df.columns)}")

    df = df.copy()
    df["text"] = df[text_column].astype(str).str.strip()
    df["sentiment_score"] = df[score_column].astype(float)
    df["label"] = df["sentiment_score"].apply(_label_from_score).astype(int)

    has_split_col = "split" in df.columns
    if not has_split_col:
        df["monologue_idx"] = df[id_column].apply(_normalize_video_index)
        if df["monologue_idx"].isna().any():
            raise ValueError(
                "CMU-MOSI rows have non-integer video IDs, but no 'split' column "
                "is present. Either add a 'split' column (train/val/test) or "
                "remap your video IDs to integer monologue indices 1..93 "
                "(see scripts/preprocess_mosi.py for guidance)."
            )
        df["split"] = df["monologue_idx"].astype(int).apply(
            lambda i: _split_for_monologue(i, train_monologues, val_monologues, test_monologues)
        )
        df = df[df["split"].notna()].reset_index(drop=True)

    df["utt_id"] = df[id_column].astype(str) + "_" + df.groupby(id_column).cumcount().astype(str)
    df["dialogue_id"] = df[id_column].astype(str)
    out_cols = ["utt_id", "dialogue_id", "text", "sentiment_score", "label", "split"]
    df = df[out_cols]

    return (
        df[df.split == "train"].reset_index(drop=True),
        df[df.split == "val"].reset_index(drop=True),
        df[df.split == "test"].reset_index(drop=True),
    )


def write_manifests(
    csv_path: str | Path,
    manifest_dir: str | Path,
    text_column: str,
    score_column: str,
    id_column: str,
    train_monologues: Sequence[int],
    val_monologues: Sequence[int],
    test_monologues: Sequence[int],
) -> Dict[str, Path]:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = build_manifest(
        csv_path, text_column, score_column, id_column,
        train_monologues, val_monologues, test_monologues,
    )
    paths = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = manifest_dir / f"{split}.csv"
        df.to_csv(out, index=False)
        paths[split] = out
    return paths
