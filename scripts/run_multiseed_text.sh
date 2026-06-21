#!/usr/bin/env bash
# Run Stage I (fine-tune) -> extract features -> Stage II for a list of seeds.
# Re-uses the same MSP-PODCAST pre-trained RoBERTa-FT across all seeds — that
# step has tiny variance (148K training samples) so re-running it is wasteful.
#
# Usage:
#   bash scripts/run_multiseed_text.sh                 # default seeds: 1 2 3
#   bash scripts/run_multiseed_text.sh 1 2 3 7         # custom seeds
#
# Estimated time on RTX 4090: ~37 min per seed (Stage I 25m + extract 2m + Stage II 10m)
# so 3 seeds ≈ 2 hours total.

set -euo pipefail

# --- config -------------------------------------------------------------------
SEEDS="${@:-1 2 3}"
ROBERTA_FT="outputs/pretrain_msp/best"   # frozen across seeds (precondition)
STAGE1_CFG="configs/iemocap_text.yaml"
STAGE2_CFG="configs/iemocap_text_stage2.yaml"
STAGE1_LR="2.0e-5"                        # known-stable for RoBERTa-large
GPU="${CUDA_VISIBLE_DEVICES:-0}"

# --- pre-flight --------------------------------------------------------------
if [[ ! -d "${ROBERTA_FT}" ]]; then
    echo "❌ ${ROBERTA_FT} not found. Run pretrain first: python -m src.train --config configs/pretrain_msp.yaml"
    exit 1
fi
mkdir -p data/cache logs

echo "==========================================="
echo "Multi-seed Stage I + II text pipeline"
echo "  Seeds:       ${SEEDS}"
echo "  RoBERTa-FT:  ${ROBERTA_FT}"
echo "  GPU:         ${GPU}"
echo "==========================================="

# --- main loop --------------------------------------------------------------
for s in ${SEEDS}; do
    echo ""
    echo "########## seed=${s} : Stage I ##########"
    python -m src.train --config "${STAGE1_CFG}" \
        --override \
            "seed=${s}" \
            "model.pretrained=\"${ROBERTA_FT}\"" \
            "train.learning_rate=${STAGE1_LR}" \
            "run_name=stage1_seed${s}" \
            "output_dir=outputs/stage1_seed${s}"

    echo ""
    echo "########## seed=${s} : Extract Stage I features ##########"
    python -m scripts.extract_text_features \
        --stage1-ckpt "outputs/stage1_seed${s}/best" \
        --manifest-dir data/manifests/iemocap \
        --out-pt "data/cache/features_seed${s}.pt" \
        --batch-size 64

    echo ""
    echo "########## seed=${s} : Stage II ##########"
    python -m src.train_stage2 --config "${STAGE2_CFG}" \
        --override \
            "seed=${s}" \
            "dataset.features_path=\"data/cache/features_seed${s}.pt\"" \
            "run_name=stage2_seed${s}" \
            "output_dir=outputs/stage2_seed${s}"
done

echo ""
echo "==========================================="
echo "ALL DONE. Summary:"
python -m scripts.summarize_multiseed --seeds ${SEEDS}
