"""Build MELD CSV manifests.

Expects the standard MELD release with `train_sent_emo.csv`, `dev_sent_emo.csv`,
`test_sent_emo.csv`.

Usage:
    python -m scripts.preprocess_meld --meld-root /workspace/datasets/MELD
"""
from __future__ import annotations

import argparse

from src.data import meld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meld-root", required=True, type=str)
    parser.add_argument("--out-dir", default="data/manifests/meld", type=str)
    args = parser.parse_args()

    label_map = {
        "anger": 0,
        "disgust": 1,
        "fear": 2,
        "joy": 3,
        "neutral": 4,
        "sadness": 5,
        "surprise": 6,
    }
    paths = meld.write_manifests(
        root=args.meld_root,
        manifest_dir=args.out_dir,
        csv_train="train_sent_emo.csv",
        csv_val="dev_sent_emo.csv",
        csv_test="test_sent_emo.csv",
        text_column="Utterance",
        label_column="Emotion",
        label_map=label_map,
    )
    for split, p in paths.items():
        print(f"{split}: {p}")


if __name__ == "__main__":
    main()
