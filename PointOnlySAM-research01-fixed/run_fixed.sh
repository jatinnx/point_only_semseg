#!/usr/bin/env bash
set -euo pipefail

# PointOnlySAM-research01-fixed runner
# Assumes these are sibling directories/files:
#   ../point_only_sam_rs_Es_5pt/
#   ../PointSAM/
#   ../sam_vit_b_01ec64.pth

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/../.venv/bin/python" ]]; then
    PYTHON="$ROOT/../.venv/bin/python"
elif [[ -n "${PYTHON:-}" ]] && command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON="$PYTHON"
else
    PYTHON="$(command -v python3 || command -v python)"
fi

CONFIG="$ROOT/configs/dlrsd_pointonly_sam_fixed.json"
TRAIN_MANIFEST="$ROOT/../point_only_sam_rs_Es_5pt/data/train.json"
VAL_MANIFEST="$ROOT/../point_only_sam_rs_Es_5pt/data/val.json"
SAM_SOURCE="$ROOT/../PointSAM"
SAM_CHECKPOINT="$ROOT/../sam_vit_b_01ec64.pth"
CACHE="$ROOT/artifacts/dlrsd_sam_regions_fixed.pt"
CKPT="$ROOT/runs/dlrsd_pointonly_sam_fixed/last.pt"

check_paths() {
    echo "Fixed project : $ROOT"
    echo "Python        : $PYTHON"
    echo "Config        : $CONFIG"
    echo "Train manifest: $TRAIN_MANIFEST"
    echo "Val manifest  : $VAL_MANIFEST"
    echo "SAM source    : $SAM_SOURCE"
    echo "SAM checkpoint: $SAM_CHECKPOINT"
    echo "Cache         : $CACHE"
    echo "Checkpoint    : $CKPT"
    echo

    for p in "$CONFIG" "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$SAM_SOURCE" "$SAM_CHECKPOINT"; do
        if [[ ! -e "$p" ]]; then
            echo "ERROR: missing path: $p" >&2
            exit 1
        fi
    done
    echo "All required paths exist."
}

case "${1:-check}" in
    check)
        check_paths
        ;;
    cache)
        check_paths
        "$PYTHON" build_fixed_geometry_cache.py \
            --config "$CONFIG" \
            --output "$CACHE"
        ;;
    smoke)
        check_paths
        if [[ ! -f "$CACHE" ]]; then
            echo "Cache not found. Building it first..."
            "$PYTHON" build_fixed_geometry_cache.py \
                --config "$CONFIG" \
                --output "$CACHE"
        fi
        "$PYTHON" train_fixed.py \
            --config "$CONFIG" \
            --max-steps 1
        ;;
    train)
        check_paths
        if [[ ! -f "$CACHE" ]]; then
            echo "Cache not found. Building it first..."
            "$PYTHON" build_fixed_geometry_cache.py \
                --config "$CONFIG" \
                --output "$CACHE"
        fi
        "$PYTHON" train_fixed.py \
            --config "$CONFIG"
        ;;
    resume)
        check_paths
        if [[ ! -f "$CACHE" ]]; then
            echo "ERROR: cache not found: $CACHE" >&2
            exit 1
        fi
        if [[ -z "${2:-}" ]]; then
            echo "Usage: ./run_fixed.sh resume CHECKPOINT" >&2
            exit 1
        fi
        "$PYTHON" train_fixed.py \
            --config "$CONFIG" \
            --resume "$2"
        ;;
    eval)
        check_paths
        if [[ ! -f "$CKPT" ]]; then
            echo "ERROR: checkpoint not found: $CKPT" >&2
            echo "Train first with: ./run_fixed.sh train" >&2
            exit 1
        fi
        "$PYTHON" evaluate_fixed.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --mode image_only
        ;;
    eval_points)
        check_paths
        if [[ ! -f "$CKPT" ]]; then
            echo "ERROR: checkpoint not found: $CKPT" >&2
            exit 1
        fi
        "$PYTHON" evaluate_fixed.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --mode point_set
        ;;
    infer)
        check_paths
        if [[ ! -f "$CKPT" ]]; then
            echo "ERROR: checkpoint not found: $CKPT" >&2
            exit 1
        fi
        if [[ -z "${2:-}" ]]; then
            echo "Usage: ./run_fixed.sh infer IMAGE_OR_DIRECTORY [OUTPUT_DIR]" >&2
            exit 1
        fi
        OUTPUT_DIR="${3:-$ROOT/inference_fixed}"
        "$PYTHON" infer_fixed.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --input "$2" \
            --output-dir "$OUTPUT_DIR"
        ;;
    *)
        echo "Usage:"
        echo "  ./run_fixed.sh check"
        echo "  ./run_fixed.sh cache"
        echo "  ./run_fixed.sh smoke"
        echo "  ./run_fixed.sh train"
        echo "  ./run_fixed.sh resume CHECKPOINT"
        echo "  ./run_fixed.sh eval"
        echo "  ./run_fixed.sh eval_points"
        echo "  ./run_fixed.sh infer IMAGE_OR_DIRECTORY [OUTPUT_DIR]"
        exit 1
        ;;
esac
