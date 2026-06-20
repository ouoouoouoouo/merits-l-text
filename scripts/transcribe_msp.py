"""Run Whisper-large-v3 over an MSP-PODCAST audio directory.

Output: CSV with columns (utt_id, audio_path, text). Resume-friendly: re-running
appends any utts not yet in the output file.

Important: MSP-PODCAST ships ~270k .wav files, but only ~149k of them are the
labelled subset used by the paper. Pass --reference-dir pointing at
MSP_Podcast/Transcripts/ (or any folder whose .txt stems are the utt_ids you
care about) to restrict transcription to just those — saves ~5-10 hours of GPU.

Timing on RTX 4090 (faster-whisper, large-v3, batched):
    149k utts (~230h audio) → ~5-8 hours with --batch-size 16

Usage:
    python -m scripts.transcribe_msp \
        --audio-root /home/ouo/dataset/MSP_Podcast/Audios \
        --reference-dir /home/ouo/dataset/MSP_Podcast/Transcripts \
        --out-csv data/manifests/msp_podcast/transcripts.csv \
        --batch-size 16 --compute-type float16
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Set

from tqdm import tqdm

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")


def _list_audio_files(audio_root: Path) -> List[Path]:
    return sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def _load_reference_ids(reference_dir: Optional[Path]) -> Optional[Set[str]]:
    """Read utt_ids from a reference folder (matches by filename stem)."""
    if reference_dir is None:
        return None
    if not reference_dir.is_dir():
        raise FileNotFoundError(f"--reference-dir not found: {reference_dir}")
    ids = {p.stem for p in reference_dir.iterdir() if p.is_file()}
    if not ids:
        raise RuntimeError(f"No files under {reference_dir}; cannot build reference id set")
    return ids


def _read_done(out_csv: Path) -> Set[str]:
    if not out_csv.exists():
        return set()
    done: Set[str] = set()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("utt_id"):
                done.add(r["utt_id"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--reference-dir", default=None, type=str,
                        help="Only transcribe files whose stem appears in this directory. "
                             "Use for the 149k labelled subset by passing MSP_Podcast/Transcripts/")
    parser.add_argument("--model", default="large-v3", type=str,
                        help="faster-whisper model id (large-v3, large-v3-turbo, medium, etc.)")
    parser.add_argument("--language", default="en", type=str)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--compute-type", default="float16",
                        help="float16 / int8_float16 / float32")
    parser.add_argument("--batch-size", default=16, type=int,
                        help="faster-whisper BatchedInferencePipeline batch size. "
                             "16 fits on 24GB GPU (4090) with large-v3.")
    parser.add_argument("--beam-size", default=1, type=int)
    parser.add_argument("--vad-filter", action="store_true",
                        help="Enable Silero VAD pre-filter (faster on long files with silence).")
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ref_ids = _load_reference_ids(Path(args.reference_dir)) if args.reference_dir else None
    done_ids = _read_done(out_csv)

    # Enumerate & filter
    all_files = _list_audio_files(audio_root)
    if not all_files:
        raise FileNotFoundError(f"No audio under {audio_root}")
    if ref_ids is not None:
        files = [p for p in all_files if p.stem in ref_ids and p.stem not in done_ids]
        print(f"Audio total {len(all_files)}, reference {len(ref_ids)}, "
              f"already done {len(done_ids)}, to process {len(files)}")
    else:
        files = [p for p in all_files if p.stem not in done_ids]
        print(f"Audio total {len(all_files)}, already done {len(done_ids)}, "
              f"to process {len(files)}")
    if not files:
        print("Nothing to do.")
        return

    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError as e:
        raise SystemExit("`pip install faster-whisper` first") from e

    print(f"Loading {args.model} on {args.device} ({args.compute_type})...")
    base = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    pipe = BatchedInferencePipeline(model=base)

    # Append mode so resume works.
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    fout = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["utt_id", "audio_path", "text"])

    try:
        for path in tqdm(files, desc="transcribe"):
            try:
                segments, _ = pipe.transcribe(
                    str(path),
                    language=args.language,
                    beam_size=args.beam_size,
                    batch_size=args.batch_size,
                    vad_filter=args.vad_filter,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {path.name}: {e}")
                continue
            writer.writerow([path.stem, str(path), text])
            fout.flush()
    finally:
        fout.close()

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
