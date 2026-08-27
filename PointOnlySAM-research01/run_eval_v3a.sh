#!/bin/bash
set -euo pipefail
cd /home/cse-sdpl/Downloads/point_only_semseg/PointOnlySAM-research01
../.venv/bin/python -u eval_visualize.py \
  --config configs/dlrsd_pointonly_sam_v3a_lora.json \
  --checkpoint runs/dlrsd_pointonly_sam_v3a_lora/last.pt \
  --output-dir runs/dlrsd_pointonly_sam_v3a_lora/eval_last01 \
  2>&1 | tee eval_v3a.log
