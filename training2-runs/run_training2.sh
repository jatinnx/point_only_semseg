#!/usr/bin/env bash
set -euo pipefail

# Training2 run - separate from original PointOnlySAM-research01-fixed
# This script trains and then evaluates without modifying any original files

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT/.." && pwd)"

# Python
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

# Paths
CONFIG="$ROOT/config_training2.json"
LOG_DIR="$ROOT/logs"
CKPT_DIR="$ROOT/checkpoints"
PRED_DIR="$ROOT/predictions"
EVAL_DIR="$ROOT/evals"
TRAIN_SCRIPT="$PROJECT_ROOT/PointOnlySAM-research01-fixed/train_fixed.py"
EVAL_SCRIPT="$PROJECT_ROOT/PointOnlySAM-research01-fixed/evaluate_fixed.py"

# Create directories
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$PRED_DIR" "$EVAL_DIR"

echo "=========================================="
echo "Training2 Run Started at $(date)"
echo "=========================================="
echo "Config: $CONFIG"
echo "Checkpoints: $CKPT_DIR"
echo "Logs: $LOG_DIR"
echo "Predictions: $PRED_DIR"
echo "Evaluations: $EVAL_DIR"
echo ""

# Run training
echo "[1/2] Starting training (40 epochs, ~4.6 hours estimated)..."
"$PYTHON" "$TRAIN_SCRIPT" --config "$CONFIG" 2>&1 | tee "$LOG_DIR/training2_train.log"
TRAIN_EXIT=$?

if [ $TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "=========================================="
    echo "ERROR: Training failed with exit code $TRAIN_EXIT"
    echo "Check log: $LOG_DIR/training2_train.log"
    echo "=========================================="
    exit $TRAIN_EXIT
fi

echo ""
echo "[2/2] Training complete! Running evaluation..."

# Find the last checkpoint
CKPT="$CKPT_DIR/last.pt"
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: No checkpoint found at $CKPT"
    exit 1
fi

# Run evaluation (image_only mode)
"$PYTHON" "$EVAL_SCRIPT" \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --mode image_only \
    --save-preds "$PRED_DIR/eval_last" \
    2>&1 | tee "$LOG_DIR/training2_eval.log"

echo ""
echo "=========================================="
echo "Training2 Run Completed at $(date)"
echo "=========================================="
echo "Checkpoints: $CKPT_DIR"
echo "Training log: $LOG_DIR/training2_train.log"
echo "Evaluation log: $LOG_DIR/training2_eval.log"
echo "Predictions: $PRED_DIR/eval_last"
echo ""
