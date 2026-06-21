#!/usr/bin/env bash
# Hyperparameter sweep for Stage II text. All runs use seed=42 + the same
# Stage I features (data/cache/iemocap_text_features.pt) so the only thing
# varying is the model / training hparams.
#
# Each run ≈ 10 minutes on RTX 4090. 5 configs ≈ 50 minutes.
#
# Usage:
#   bash scripts/sweep_stage2.sh
#
# After it finishes, scripts/summarize_multiseed.py can compare:
#   python -m scripts.summarize_multiseed --seeds 42

set -euo pipefail

CFG="configs/iemocap_text_stage2.yaml"
FEATURES="data/cache/iemocap_text_features.pt"

if [[ ! -f "${FEATURES}" ]]; then
    echo "❌ ${FEATURES} not found. Run extract_text_features first (e.g. via run_multiseed_text.sh)."
    exit 1
fi

run_one() {
    # args: tag, override pairs ...
    local tag="$1"
    shift
    local out="outputs/stage2_${tag}"
    echo ""
    echo "############# sweep: ${tag} #############"
    python -m src.train_stage2 --config "${CFG}" \
        --override \
            "seed=42" \
            "dataset.features_path=\"${FEATURES}\"" \
            "run_name=stage2_${tag}" \
            "output_dir=${out}" \
            "$@"
}

# baseline reference: gru_hidden=256, num_heads=4, layers=1, dropout=0.3, lr=1e-4
# (you already have this as stage2_seed42 → 0.8196)

# A) bigger model
run_one "A_big" \
    "model.gru_hidden=512" \
    "model.num_heads=8" \
    "model.gru_layers=1" \
    "model.dropout=0.3"

# B) bigger + deeper + more reg
run_one "B_big_deep" \
    "model.gru_hidden=512" \
    "model.num_heads=8" \
    "model.gru_layers=2" \
    "model.dropout=0.5"

# C) deeper but same width, more reg
run_one "C_deep" \
    "model.gru_hidden=256" \
    "model.num_heads=4" \
    "model.gru_layers=2" \
    "model.dropout=0.5"

# D) even bigger, longer training, smaller lr
run_one "D_huge_long" \
    "model.gru_hidden=512" \
    "model.num_heads=8" \
    "model.gru_layers=2" \
    "model.dropout=0.5" \
    "train.epochs=100" \
    "train.learning_rate=5.0e-5" \
    "train.early_stopping_patience=20"

# E) regularization heavy, same size
run_one "E_reg" \
    "model.gru_hidden=256" \
    "model.num_heads=4" \
    "model.gru_layers=1" \
    "model.dropout=0.5" \
    "train.epochs=100" \
    "train.early_stopping_patience=15"

echo ""
echo "==========================================="
echo "SWEEP DONE. Final test/weighted_f1 per config:"
python - <<'PY'
import json
from pathlib import Path

rows = []
for d in sorted(Path("outputs").glob("stage2_*")):
    # skip the multiseed runs (stage2_seed*) — only show sweep tags
    if d.name.startswith("stage2_seed"):
        continue
    mpath = d / "metrics.jsonl"
    if not mpath.exists():
        continue
    best = None
    with mpath.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("prefix") == "test" and "weighted_f1" in r:
                best = float(r["weighted_f1"])
    rows.append((d.name, best))

# baseline reference
seed42 = Path("outputs/stage2_seed42/metrics.jsonl")
if seed42.exists():
    best = None
    with seed42.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("prefix") == "test" and "weighted_f1" in r:
                best = float(r["weighted_f1"])
    rows.append(("stage2_seed42 (baseline)", best))

rows.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0)))
print(f"{'run':<35} | wF1")
print("-" * 50)
for name, wf1 in rows:
    val = f"{wf1:.4f}" if wf1 is not None else "MISSING"
    print(f"{name:<35} | {val}")
PY
