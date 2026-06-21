"""Print mean ± std of test weighted_f1 across seeds for Stage I and Stage II.

Reads `outputs/{stage1,stage2}_seed{S}/metrics.jsonl` (we look for the last
line with `prefix=test`).

Usage:
    python -m scripts.summarize_multiseed --seeds 1 2 3 42
    python -m scripts.summarize_multiseed                       # auto-detect all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _last_test_metric(metrics_jsonl: Path, key: str = "weighted_f1") -> float | None:
    if not metrics_jsonl.exists():
        return None
    best = None
    with metrics_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prefix") == "test" and key in row:
                best = float(row[key])  # take last (final)
    return best


def _discover_seeds(stage: str) -> List[int]:
    pat = re.compile(rf"^{re.escape(stage)}_seed(\d+)$")
    seeds = []
    for d in Path("outputs").glob(f"{stage}_seed*"):
        m = pat.match(d.name)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(set(seeds))


def _gather(stage: str, seeds: List[int], key: str) -> List[Tuple[int, float]]:
    out = []
    for s in seeds:
        m = _last_test_metric(Path("outputs") / f"{stage}_seed{s}" / "metrics.jsonl", key)
        if m is not None:
            out.append((s, m))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="If omitted, auto-detect all seeds under outputs/.")
    parser.add_argument("--key", default="weighted_f1", type=str,
                        help="Metric name (default weighted_f1).")
    args = parser.parse_args()

    seeds = args.seeds or sorted(set(_discover_seeds("stage1") + _discover_seeds("stage2")))
    if not seeds:
        print("No seeded runs found under outputs/. Did you finish run_multiseed_text.sh?")
        return

    for stage in ("stage1", "stage2"):
        results = _gather(stage, seeds, args.key)
        if not results:
            print(f"\n[{stage.upper()}] no test metrics found.")
            continue
        vals = np.array([v for _, v in results])
        print(f"\n[{stage.upper()}] test/{args.key} across seeds")
        for s, v in results:
            print(f"  seed={s:>3}: {v:.4f}")
        print(f"  ─────────────────────")
        print(f"  mean = {vals.mean():.4f}")
        print(f"  std  = {vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")
        print(f"  ==> {vals.mean():.4f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}")


if __name__ == "__main__":
    main()
