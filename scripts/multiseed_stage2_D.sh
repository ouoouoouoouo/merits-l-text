#!/usr/bin/env bash
# Multi-seed verification of the best sweep config (D_huge_long).
# Uses the same Stage I features cache; only Stage II hparams + seed vary.
#
# Estimated: ~20 min/seed on RTX 4090. 4 seeds ≈ 80 min.

set -euo pipefail

SEEDS="${@:-1 2 3 42}"
CFG="configs/iemocap_text_stage2.yaml"
FEATURES="data/cache/iemocap_text_features.pt"

if [[ ! -f "${FEATURES}" ]]; then
    echo "❌ ${FEATURES} not found. Run extract_text_features first."
    exit 1
fi

echo "==========================================="
echo "Stage II 'D' config multi-seed verification"
echo "  Config: hidden=512, 2-layer, 8 heads, dropout=0.5, lr=5e-5, 100 ep"
echo "  Seeds:  ${SEEDS}"
echo "==========================================="

for s in ${SEEDS}; do
    echo ""
    echo "########## D + seed=${s} ##########"
    python -m src.train_stage2 --config "${CFG}" \
        --override \
            "seed=${s}" \
            "model.gru_hidden=512" \
            "model.gru_layers=2" \
            "model.num_heads=8" \
            "model.dropout=0.5" \
            "train.learning_rate=5.0e-5" \
            "train.epochs=100" \
            "train.early_stopping_patience=20" \
            "dataset.features_path=\"${FEATURES}\"" \
            "run_name=stage2_D_seed${s}" \
            "output_dir=outputs/stage2_D_seed${s}"
done

echo ""
echo "==========================================="
echo "COMPARISON: baseline (stage2_seed*) vs D config (stage2_D_seed*)"
python - <<'PY'
import json
from pathlib import Path
import numpy as np

def last_test(p: Path, key="weighted_f1"):
    if not p.exists():
        return None
    best = None
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("prefix") == "test" and key in r:
                best = float(r[key])
    return best

def gather(prefix):
    out = []
    for d in sorted(Path("outputs").glob(f"{prefix}_seed*")):
        # skip the D ones when looking for baseline
        if prefix == "stage2" and "_D_" in d.name:
            continue
        m = last_test(d / "metrics.jsonl")
        if m is not None:
            seed = d.name.split("seed")[-1]
            out.append((int(seed), m))
    return sorted(out)

baseline = gather("stage2")
d_runs = []
for d in sorted(Path("outputs").glob("stage2_D_seed*")):
    m = last_test(d / "metrics.jsonl")
    if m is not None:
        seed = int(d.name.split("seed")[-1])
        d_runs.append((seed, m))
d_runs.sort()

def show(name, runs):
    if not runs:
        print(f"  [{name}] no runs found.")
        return None
    vals = np.array([v for _, v in runs])
    print(f"  [{name}]  ({len(runs)} seeds)")
    for s, v in runs:
        print(f"    seed={s:>3}: {v:.4f}")
    mean = vals.mean()
    std = vals.std(ddof=1) if len(vals) > 1 else 0.0
    print(f"    mean ± std = {mean:.4f} ± {std:.4f}")
    return mean, std, vals

print()
b = show("baseline", baseline)
print()
d = show("D config", d_runs)
print()
if b and d:
    bm, bs, bv = b
    dm, ds, dv = d
    print("─" * 50)
    print(f"  Δ mean = {dm-bm:+.4f}")
    if len(bv) > 1 and len(dv) > 1:
        from scipy import stats as scipy_stats  # type: ignore
        try:
            t, p = scipy_stats.ttest_ind(dv, bv, equal_var=False)
            print(f"  t-test (Welch): t={t:+.3f}, p={p:.3f}")
            if p < 0.05:
                print("  → D is statistically significantly different from baseline (p<0.05).")
            else:
                print("  → D's gain is NOT statistically significant; within seed noise.")
        except ImportError:
            print("  (scipy not installed; skipping t-test)")
PY
