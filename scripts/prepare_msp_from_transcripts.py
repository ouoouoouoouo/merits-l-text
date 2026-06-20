"""Convert MSP-PODCAST official transcripts/ folder into a CSV manifest.

Use this when the dataset already ships with ground-truth transcripts and you
want to skip the Whisper ASR step (recommended — official transcripts are
cleaner than Whisper-large-v3 output, so the LLM labels will be more accurate).

Input  : /path/to/MSP_Podcast/Transcripts/MSP-PODCAST_XXXX_YYYY.txt
Output : data/manifests/msp_podcast/transcripts.csv  with columns (utt_id, text)

Usage:
    python -m scripts.prepare_msp_from_transcripts \
        --transcripts-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
        --out-csv data/manifests/msp_podcast/transcripts.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-dir", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--min-chars", default=2, type=int,
                        help="Drop transcripts shorter than this many characters (likely garbage).")
    parser.add_argument("--max-chars", default=2000, type=int,
                        help="Drop transcripts longer than this (likely long monologues that "
                             "would waste GPT-3.5 tokens; the paper works on utterance-level).")
    args = parser.parse_args()

    tdir = Path(args.transcripts_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(tdir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt under {tdir}")
    print(f"Found {len(files)} transcript files")

    n_written, n_empty, n_short, n_long = 0, 0, 0, 0
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "text"])
        for path in tqdm(files, desc="converting"):
            try:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {path.name}: {e}")
                continue
            # Collapse internal whitespace / newlines.
            text = " ".join(text.split())
            if not text:
                n_empty += 1
                continue
            if len(text) < args.min_chars:
                n_short += 1
                continue
            if len(text) > args.max_chars:
                n_long += 1
                continue
            utt_id = path.stem  # MSP-PODCAST_XXXX_YYYY
            writer.writerow([utt_id, text])
            n_written += 1

    print(
        f"Wrote {n_written} rows to {out_csv}\n"
        f"  dropped: empty={n_empty}, too_short={n_short}, too_long={n_long}"
    )


if __name__ == "__main__":
    main()
