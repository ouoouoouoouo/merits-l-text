"""Label MSP-PODCAST transcripts with GPT-3.5 Turbo (positive/negative/neutral).

The exact prompt from the paper (Sec. IV-B):

    You are a sentiment classification bot. Given the [sentence], classify
    as positive, negative or neutral sentiment. Please give the sentiment
    and no extra text as output.

This script reads `transcripts.csv` (utt_id, audio_path, text) and writes
`pseudo_labels.csv` (utt_id, text, label) where label ∈ {positive, negative, neutral}.

Cost estimate: ~149k requests, each ~50-80 input tokens + 1-2 output tokens.
At gpt-3.5-turbo current prices this is roughly $1-2 total for the full dataset.

Usage:
    python -m scripts.llm_pseudo_label_msp \
        --in-csv data/manifests/msp_podcast/transcripts.csv \
        --out-csv data/manifests/msp_podcast/pseudo_labels.csv \
        --model gpt-3.5-turbo \
        --concurrency 16

Requires OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

from tqdm.asyncio import tqdm_asyncio

PROMPT = (
    "You are a sentiment classification bot. Given the [sentence], classify as "
    "positive, negative or neutral sentiment. Please give the sentiment and no "
    "extra text as output."
)

VALID = {"positive", "negative", "neutral"}


def _parse_label(text: str) -> str | None:
    """Return one of VALID, else None."""
    t = text.strip().lower().strip(".!,\"' ")
    if t in VALID:
        return t
    # Sometimes models echo: "Sentiment: positive"
    for v in VALID:
        if v in t:
            return v
    return None


async def _label_one(client, model: str, utt_id: str, text: str, semaphore) -> Tuple[str, str, str | None]:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"[sentence]: {text}"},
                ],
            )
            raw = resp.choices[0].message.content or ""
            return utt_id, text, _parse_label(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {utt_id}: {e}", file=sys.stderr)
            return utt_id, text, None


def _read_existing(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    done = set()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["utt_id"])
    return done


async def _main_async(args) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise SystemExit("`pip install openai>=1.30` first") from e

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Read inputs
    rows: List[Tuple[str, str]] = []
    with in_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            text = (r.get("text") or "").strip()
            if not text:
                continue
            rows.append((r["utt_id"], text))

    done = _read_existing(out_csv)
    pending = [(u, t) for u, t in rows if u not in done]
    print(f"Total {len(rows)}, already labelled {len(done)}, to process {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(int(args.concurrency))

    # Resume-friendly: open in append mode, write header only if new.
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    fout = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["utt_id", "text", "label"])

    try:
        tasks = [_label_one(client, args.model, u, t, sem) for u, t in pending]
        # Use as_completed-style to flush progressively.
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            utt_id, text, label = await fut
            if label is None:
                continue
            writer.writerow([utt_id, text, label])
            fout.flush()
    finally:
        fout.close()

    print(f"Wrote labels to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--concurrency", default=16, type=int)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
