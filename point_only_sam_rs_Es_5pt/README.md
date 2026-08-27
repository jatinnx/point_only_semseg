# point_only_sam_rs_Es_5pt — Point-only SAM, v2 (clean-room)

**Semantic segmentation on DLRSD with ZERO dense labels in training** — the
only supervision is 5 points per class per image, taken from **the Chakraborty
paper's own point annotations** (`dlrsd/point_1cmasks/`). This codebase is a
clean rebuild of `point_only_sam_rs/` with every known failure mode fixed up
front (see [newidea.txt](../../newidea.txt) for the full design rationale).

## Why this codebase exists

The v1 codebase (`point_only_sam_rs/`) reached E1 mIoU 0.5188 with zero dense
labels (~80% of Chakraborty's 64.86% mixed-supervision headline) but its
prototype experiments (E2) failed — and an audit found nine root causes, all
addressed here:

| # | v1 problem | v2 fix |
|---|---|---|
| 1 | Colour palette duplicated & mismatched between prediction PNGs and legend | **`data/class_map.py`** is the single source of truth; `colors.py`, `make_manifests.py` and `evaluate.py` all import from it |
| 2 | Prototypes frozen at epoch-0 encoder features, went stale | **Live bank**: updated EVERY step from the model's own high-confidence predictions + full refresh every `proto_refresh_every` epochs |
| 3 | `refine_consistency` KL loss contributed <1% (dead) | **Removed entirely**; replaced by `cosine_reg_self` on live centroids |
| 4 | `refine()` at eval time corrupted predictions | **Off by default** (`proto_use_refine_at_eval=False`); only via `--use-refine` |
| 5 | No prototype refresh mechanism in E2 | `update_from_predictions()` every step + `refresh()` every N epochs |
| 6 | 16% of sampled points on class boundaries | Points come from **Chakraborty's interior-only annotations** (0.00% boundary, verified) |
| 7 | Near-duplicate point groups (≤5px apart) | **Grid selection** — 5 points/class spread across a 3×3 cell grid |
| 8 | No E0 dense baseline | Legacy codebase remains the dense reference; E1..E8 use the same split/protocol |
| 9 | `gate_px=0` forever, no prototype diagnostics | **Rich per-step logging**: point_CE / proto_reg / bank pixels / per-class top-1 cosine |

## Layout

```
point_only_sam_rs_Es_5pt/
├── data/
│   ├── class_map.py        ← THE class names + palette (single source of truth)
│   ├── make_manifests.py   ← reads Chakraborty point_1cmasks, 5 pts/class via grid
│   ├── dataset.py          ← point-only contract (mask key forbidden in train)
│   ├── train.json / val.json / val_masks_remapped/
├── model/
│   ├── lora.py             ← LoRALinear, inject_lora, EMA teacher (unchanged, correct)
│   ├── decoder.py          ← SemanticDecoder (unchanged, correct)
│   └── sam_wrapper.py      ← PointOnlySAM: ONE encode() per step, shared embedding
├── core/
│   ├── prototypes.py       ← LIVE bank: init + self-update every step + refresh
│   ├── pseudo.py           ← per-pixel gate + fusion (E3+)
│   ├── losses.py           ← point CE, pseudo CE, consistency, smoothness (no refine_consistency)
│   ├── prompts.py          ← NegativePromptSampler (correct)
│   └── colors.py           ← colorize/overlay/legend — all from class_map.PALETTE
├── configs/                base.py + e1..e8 (one flag apart)
├── train.py  evaluate.py  run_experiment.py
└── runs/                   checkpoints/ logs/ predictions/ (artifact map below)
```

## Experiment ladder (E1..E8)

| Exp | Adds | Key question |
|---|---|---|
| E1 | SAM + LoRA + point CE | How much can SAM learn from 5 px/class? |
| E2 | **live prototype bank** (self-update + cosine reg) | Does a fresh bank beat E1? |
| E3 | EMA teacher + gated pseudo-labels + SAM masks | Does dense pseudo-labelling help? |
| E4 | FIFO bank growth from teacher pseudo pixels | Do better prototypes improve E3? |
| E5 | negative SAM prompts | Separate spectrally-confused classes |
| E6 | 3-component confidence gate | Smarter pseudo-label filtering |
| E7 | image-aware smoothness loss | Sharper boundaries |
| E8 | all of the above | Full system ceiling |

## Running

```bash
# from the repo root (point_only_sam_rs_Es_5pt/ is a package)

# one-time: build manifests from Chakraborty's points (5/class, grid spread)
.venv/bin/python -m point_only_sam_rs_Es_5pt.data.make_manifests

# train E1 (50 epochs, ckpt every 5)
setsid bash -c 'cd /home/cse-sdpl/Downloads/point_only_semseg && \
  .venv/bin/python -u -m point_only_sam_rs_Es_5pt.run_experiment \
    --config /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/configs/e1_point_only.py \
    --save-dir /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/checkpoints \
    --log /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/logs/E1_train.log \
    > /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/logs/E1_train_stdout.log 2>&1'

# evaluate a checkpoint on the full 1,319-image test set
.venv/bin/python -u -m point_only_sam_rs_Es_5pt.run_experiment --evaluate \
  --config /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/configs/e1_point_only.py \
  --checkpoint <abs path to E1_epoch_XXXX.pt> \
  --save-preds /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/predictions/E1_epXX \
  --log /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/logs/E1_epXX_eval.log
```

## House rules (same as v1 — they were correct)

- **Use ABSOLUTE paths** for `--save-dir`/`--log`/`--save-preds` (the
  `resolve()` helper double-anchors repo-root-relative CLI args into nested
  dirs).
- Detach with `setsid bash -c '…' &` (plain `nohup` dies with the shell).
- Checkpoints: `runs/checkpoints/<EXP>_epoch_XXXX.pt` (~383 MB each).
- Every eval writes metrics + per-class IoU to `runs/logs/<EXP>_ep<N>_eval.log`.

## Status

- Manifests built from Chakraborty points: 630 train items, 10,310 points,
  **0.00% boundary** (class 13 "ship" has 0 training points in the paper's
  annotations — a documented limitation of using the exact paper supervision).
- Smoke tests pass (E1 and E2 train; prototype init, self-update, refresh and
  eval all verified). Results progress in `runs/README.md` once full runs land.
