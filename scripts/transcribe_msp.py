"""Run Whisper-large-v3 over an MSP-PODCAST audio directory.

Output: CSV with columns (utt_id, audio_path, text).

The full MSP-PODCAST dataset is ~149k utterances. On an A100 with batched
faster-whisper you should expect ~6-10 hours total; on Whisper-large-v3
non-batched, considerably longer. Tune --batch-size / --workers accordingly.

Usage:
    python -m scripts.transcribe_msp \
        --audio-root /workspace/datasets/MSP-PODCAST/Audio \
        --out-csv data/manifests/msp_podcast/transcripts.csv \
        --model openai/whisper-large-v3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from tqdm import tqdm


def _list_audio_files(audio_root: Path, exts=(".wav", ".flac", ".mp3", ".m4a")) -> List[Path]:
    return sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in exts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--model", default="openai/whisper-large-v3", type=str)
    parser.add_argument("--language", default="en", type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16", help="float16 / int8_float16 / float32")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument(
        "--backend",
        default="faster",
        choices=["faster", "hf"],
        help="`faster` uses faster-whisper (recommended). `hf` uses transformers' Whisper pipeline.",
    )
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    files = _list_audio_files(audio_root)
    if not files:
        raise FileNotFoundError(f"No audio under {audio_root}")
    print(f"Transcribing {len(files)} files with {args.backend} / {args.model}")

    if args.backend == "faster":
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise SystemExit(
                "faster-whisper not installed. `pip install faster-whisper` or pass --backend hf"
            ) from e
        model_name = args.model.split("/")[-1]  # faster-whisper accepts 'large-v3'
        model = WhisperModel(model_name, device=args.device, compute_type=args.compute_type)

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["utt_id", "audio_path", "text"])
            for path in tqdm(files):
                segs, _ = model.transcribe(str(path), language=args.language, beam_size=1)
                text = " ".join(seg.text.strip() for seg in segs).strip()
                writer.writerow([path.stem, str(path), text])
    else:
        from transformers import pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=args.model,
            device=0 if args.device == "cuda" else -1,
        )
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["utt_id", "audio_path", "text"])
            for i in tqdm(range(0, len(files), args.batch_size)):
                batch = files[i : i + args.batch_size]
                outputs = pipe([str(p) for p in batch], batch_size=args.batch_size)
                for path, out in zip(batch, outputs):
                    writer.writerow([path.stem, str(path), out["text"].strip()])

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
