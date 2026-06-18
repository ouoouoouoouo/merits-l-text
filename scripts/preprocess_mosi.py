"""Build CMU-MOSI CSV manifests.

CMU-MOSI ships in several forms; this script expects a single CSV with the
following columns (rename your own with --text-column / --score-column / --id-column):

    video_id    integer monologue index (1..93) OR YouTube ID
    text        the transcript of the clip
    sentiment   float in [-3, 3] (continuous sentiment intensity)

If `video_id` is a YouTube ID rather than an integer index, add a `split`
column in your source CSV (values: train/val/test) and the script will use it
directly. Otherwise it splits via the paper's monologue ranges.

Usage:
    python -m scripts.preprocess_mosi \
        --csv-path /workspace/datasets/CMU-MOSI/mosi.csv \
        --out-dir data/manifests/mosi
"""
from __future__ import annotations

import argparse

from src.data import mosi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", required=True, type=str)
    parser.add_argument("--out-dir", default="data/manifests/mosi", type=str)
    parser.add_argument("--text-column", default="text", type=str)
    parser.add_argument("--score-column", default="sentiment", type=str)
    parser.add_argument("--id-column", default="video_id", type=str)
    args = parser.parse_args()

    paths = mosi.write_manifests(
        csv_path=args.csv_path,
        manifest_dir=args.out_dir,
        text_column=args.text_column,
        score_column=args.score_column,
        id_column=args.id_column,
        train_monologues=[1, 49],
        val_monologues=[50, 62],
        test_monologues=[63, 93],
    )
    for split, p in paths.items():
        print(f"{split}: {p}")


if __name__ == "__main__":
    main()
