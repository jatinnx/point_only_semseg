#!/bin/bash
set -euo pipefail
cd /home/cse-sdpl/Downloads/point_only_semseg/PointOnlySAM-research01
../.venv/bin/python -u train.py --config configs/dlrsd_pointonly_sam_v3a_lora.json 2>&1 | tee training_v3a.log
