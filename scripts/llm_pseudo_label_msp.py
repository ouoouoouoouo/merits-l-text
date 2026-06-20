"""Label MSP-PODCAST transcripts with GPT-3.5 Turbo (positive/negative/neutral).

The exact prompt from the paper (Sec. IV-B):

    You are a sentiment classification bot. Given the [sentence], classify
    as positive, negative or neutral sentiment. Please give the sentiment
    and no extra text as output.

This script reads `transcripts.csv` (utt_id, [audio_path], text) and writes
`pseudo_labels.csv` (utt_id, text, label) where label ∈ {positive, negative, neutral}.

Cost estimate: ~149k requests, each ~75 input tokens + ~2 output tokens.
At gpt-3.5-turbo current prices this is roughly $5-7 total for the full dataset.

Usage:
    python -m scripts.llm_pseudo_label_msp \
        --in-csv data/manifests/msp_podcast/transcripts.csv \
        --out-csv data/manifests/msp_podcast/pseudo_labels.csv \
        --model gpt-3.5-turbo \
        --concurrency 32 \
        --wandb  # optional, requires WANDB_API_KEY

Requires OPENAI_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm.asyncio import tqdm_asyncio


class AsyncRateLimiter:
    """Hard cap on requests per second across all coroutines.

    Implements a simple sliding-window: each acquire() waits until at least
    `min_interval` has passed since the previous one. Lockless contention is
    fine because the critical section is microseconds.
    """

    def __init__(self, max_per_second: float) -> None:
        self.min_interval = 1.0 / max_per_second if max_per_second > 0 else 0.0
        self._next_ok_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_ok_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_ok_at = max(now, self._next_ok_at) + self.min_interval

PROMPT = (
    "You are a sentiment classification bot. Given the [sentence], classify as "
    "positive, negative or neutral sentiment. Please give the sentiment and no "
    "extra text as output."
)

VALID = {"positive", "negative", "neutral"}

# gpt-3.5-turbo pricing as of 2025/2026 (USD per 1M tokens).
# Override via --price-input / --price-output if pricing changes.
DEFAULT_PRICE_INPUT_PER_1M = 0.50
DEFAULT_PRICE_OUTPUT_PER_1M = 1.50


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


async def _label_one(
    client,
    model: str,
    utt_id: str,
    text: str,
    semaphore,
    rate_limiter: AsyncRateLimiter,
    max_retries: int = 6,
) -> Tuple[str, str, Optional[str], int, int]:
    """Returns (utt_id, text, label_or_None, prompt_tokens, completion_tokens).

    Retries on 429 (rate limit) and 5xx with exponential backoff + jitter.
    """
    async with semaphore:
        for attempt in range(max_retries):
            await rate_limiter.acquire()
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
                usage = getattr(resp, "usage", None)
                pt = int(getattr(usage, "prompt_tokens", 0) or 0)
                ct = int(getattr(usage, "completion_tokens", 0) or 0)
                return utt_id, text, _parse_label(raw), pt, ct
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                # Retry on rate limit (429) and transient server errors (5xx).
                retriable = ("429" in msg) or ("rate_limit" in msg.lower()) or \
                            ("500" in msg) or ("502" in msg) or ("503" in msg) or ("504" in msg) or \
                            ("timeout" in msg.lower())
                if retriable and attempt < max_retries - 1:
                    # exponential backoff: 1, 2, 4, 8, 16, 32 sec  + jitter
                    sleep = (2 ** attempt) + random.uniform(0, 1)
                    await asyncio.sleep(sleep)
                    continue
                print(f"[warn] {utt_id}: {e}", file=sys.stderr)
                return utt_id, text, None, 0, 0
        return utt_id, text, None, 0, 0


def _read_existing(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    done = set()
    with out_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["utt_id"])
    return done


def _init_wandb(args, total: int, already_done: int, to_process: int):
    """Optional WandB init. Returns the wandb module or None."""
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[warn] --wandb requested but `wandb` not installed; skipping.", file=sys.stderr)
        return None

    config = {
        "task": "llm_pseudo_labeling",
        "model": args.model,
        "concurrency": args.concurrency,
        "temperature": 0.0,
        "max_tokens": 4,
        "prompt_version": "paper_sec_IV_B",
        "prompt": PROMPT,
        "in_csv": str(args.in_csv),
        "out_csv": str(args.out_csv),
        "total_inputs": total,
        "already_done": already_done,
        "to_process": to_process,
        "price_input_per_1m": args.price_input,
        "price_output_per_1m": args.price_output,
    }
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"label_{Path(args.in_csv).stem}",
            config=config,
            reinit=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] wandb.init failed ({e}); continuing without WandB.\n"
              f"        Hint: run `wandb login` or set WANDB_API_KEY.", file=sys.stderr)
        return None
    print(f"WandB run: {wandb.run.url}")
    return wandb


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

    wb = _init_wandb(args, total=len(rows), already_done=len(done), to_process=len(pending))

    if not pending:
        print("Nothing to do.")
        if wb is not None:
            wb.finish()
        return

    client = AsyncOpenAI(timeout=30.0)
    sem = asyncio.Semaphore(int(args.concurrency))
    # OpenAI Tier 1 gpt-3.5-turbo = 500 RPM = 8.33 req/s. Default to 7.5 to
    # leave headroom; bump with --rpm if you've moved to a higher tier.
    rate_limiter = AsyncRateLimiter(max_per_second=args.rpm / 60.0)

    # Resume-friendly: open in append mode, write header only if new.
    write_header = not out_csv.exists() or out_csv.stat().st_size == 0
    fout = out_csv.open("a", encoding="utf-8", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["utt_id", "text", "label"])

    label_counter: Counter = Counter()
    total_pt = 0
    total_ct = 0
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    log_every = max(1, int(args.log_every))

    try:
        tasks = [
            _label_one(client, args.model, u, t, sem, rate_limiter, max_retries=int(args.max_retries))
            for u, t in pending
        ]
        for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            utt_id, text, label, pt, ct = await fut
            total_pt += pt
            total_ct += ct
            if label is None:
                n_fail += 1
                continue
            writer.writerow([utt_id, text, label])
            fout.flush()
            label_counter[label] += 1
            n_ok += 1

            if wb is not None and n_ok % log_every == 0:
                elapsed = time.time() - t0
                cost = (
                    total_pt / 1_000_000 * args.price_input
                    + total_ct / 1_000_000 * args.price_output
                )
                wb.log({
                    "labelled": n_ok,
                    "failed": n_fail,
                    "throughput_per_sec": n_ok / max(elapsed, 1e-6),
                    "elapsed_sec": elapsed,
                    "tokens/prompt_total": total_pt,
                    "tokens/completion_total": total_ct,
                    "cost_usd_est": cost,
                    "dist/positive": label_counter.get("positive", 0),
                    "dist/negative": label_counter.get("negative", 0),
                    "dist/neutral": label_counter.get("neutral", 0),
                })
    finally:
        fout.close()

    elapsed = time.time() - t0
    cost = (
        total_pt / 1_000_000 * args.price_input
        + total_ct / 1_000_000 * args.price_output
    )
    print(
        f"Wrote labels to {out_csv}\n"
        f"  ok={n_ok}, failed={n_fail}, elapsed={elapsed/60:.1f} min, "
        f"throughput={n_ok/max(elapsed,1e-6):.1f} req/s\n"
        f"  tokens: prompt={total_pt}, completion={total_ct}, est. cost=${cost:.4f}\n"
        f"  distribution: {dict(label_counter)}"
    )
    if wb is not None:
        wb.log({
            "labelled": n_ok,
            "failed": n_fail,
            "throughput_per_sec": n_ok / max(elapsed, 1e-6),
            "elapsed_sec": elapsed,
            "tokens/prompt_total": total_pt,
            "tokens/completion_total": total_ct,
            "cost_usd_est": cost,
            "dist/positive": label_counter.get("positive", 0),
            "dist/negative": label_counter.get("negative", 0),
            "dist/neutral": label_counter.get("neutral", 0),
        })
        wb.summary["final_cost_usd_est"] = cost
        wb.summary["final_labelled"] = n_ok
        wb.summary["final_failed"] = n_fail
        wb.summary["final_distribution"] = dict(label_counter)
        wb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--model", default="gpt-3.5-turbo", type=str)
    parser.add_argument("--concurrency", default=16, type=int,
                        help="Max in-flight requests at once.")
    parser.add_argument("--rpm", default=450, type=float,
                        help="Hard cap on requests/minute. OpenAI Tier 1 = 500 RPM; "
                             "we default to 450 to stay safely under. Bump if tier >= 2.")
    parser.add_argument("--max-retries", default=6, type=int,
                        help="Max retries per request on 429 / 5xx / timeout.")

    # WandB (optional)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable WandB logging (needs WANDB_API_KEY).")
    parser.add_argument("--wandb-project", default="merits-l-text", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str)
    parser.add_argument("--log-every", default=500, type=int,
                        help="Push a WandB scalar log every N successful labels.")

    # Pricing for cost-estimation (override if OpenAI's price changes)
    parser.add_argument("--price-input", default=DEFAULT_PRICE_INPUT_PER_1M, type=float)
    parser.add_argument("--price-output", default=DEFAULT_PRICE_OUTPUT_PER_1M, type=float)

    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
