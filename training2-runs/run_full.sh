#!/usr/bin/env bash
set -eo pipefail
cd /home/cse-sdpl/Downloads/point_only_semseg

export PYTHONUNBUFFERED=1

echo "=== Training2 Started: $(date) ==="

.venv/bin/python -u PointOnlySAM-research01-fixed/train_fixed.py \
    --config training2-runs/config_training2.json \
    2>&1 | tee training2-runs/logs/train_full.log

echo "=== Training Finished: $(date) ==="

# Auto-evaluate after training completes
CKPT="training2-runs/checkpoints/last.pt"
if [[ -f "$CKPT" ]]; then
    echo "=== Evaluation Started: $(date) ==="
    .venv/bin/python -u PointOnlySAM-research01-fixed/evaluate_fixed.py \
        --config training2-runs/config_training2.json \
        --checkpoint "$CKPT" \
        --mode image_only \
        --save-preds training2-runs/predictions/eval_last \
        2>&1 | tee training2-runs/logs/eval_full.log
    echo "=== Evaluation Finished: $(date) ==="
else
    echo "ERROR: No checkpoint found at $CKPT — skipping eval"
fi
