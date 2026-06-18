"""Build IEMOCAP CSV manifests independently of training.

Usage:
    python -m scripts.preprocess_iemocap \
        --iemocap-root "D:/CVdataset/IEMOCAP_full_release" \
        --out-dir data/manifests/iemocap
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import iemocap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap-root", required=True, type=str)
    parser.add_argument("--out-dir", default="data/manifests/iemocap", type=str)
    args = parser.parse_args()

    label_map = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
    paths = iemocap.write_manifests(
        root=args.iemocap_root,
        manifest_dir=args.out_dir,
        label_map=label_map,
        train_sessions=[2, 3, 4],
        val_sessions=[1],
        test_sessions=[5],
    )
    for split, p in paths.items():
        print(f"{split}: {p}")
    print("Done.")


if __name__ == "__main__":
    main()
