# runs/ — v2 Experiment Artifacts (checkpoints · logs · predictions)

Everything produced by the v2 (`point_only_sam_rs_Es_5pt`) training and
evaluation lives here. One naming convention, so anyone can tell which model
is which at a glance. All experiments train on **Chakraborty's own point
annotations** (5 pts/class, grid-spread, 0% boundary) with ZERO dense labels.

## Naming convention

| Artifact | Pattern | Example |
|---|---|---|
| Model checkpoint | `<EXP>_epoch_XXXX.pt` | `E1_epoch_0010.pt` |
| Training log | `<EXP>_train.log` | `E1_train.log` |
| Eval log (metrics) | `<EXP>_ep<N>_eval.log` | `E1_ep5_eval.log` |
| Prediction images | `predictions/<EXP>_ep<N>/` | `predictions/E1_ep5/` |

`<EXP>` = E1..E8 (see the ladder table below). `<N>` = epoch. Checkpoints are
saved every 5 epochs. **Use absolute paths** when launching (see House rules).

## Directory layout

```
point_only_sam_rs_Es_5pt/runs/
├── README.md            ← you are here
├── checkpoints/         ← E1 live, E2 queued, E3–E8 future
├── logs/                ← training + eval logs (+ stdout mirrors)
└── predictions/         ← per-checkpoint prediction images
```

## The experiment ladder (v2)

| Exp | What it is | Mechanism added | Status |
|---|---|---|---|
| **E1** | Point-only baseline | SAM + LoRA + point CE (5 px/class, Chakraborty's points) | ✅ **Complete** — ep50 mIoU 0.5035 |
| **E2** | + live prototype bank | Self-update every step + cosine reg + refresh every 10 epochs | 🔄 **Training** — PID 477123 |
| E3 | + teacher–student | EMA teacher, SAM prompt masks, gated pseudo-labels | Pending |
| E4 | + proto reg from teacher | FIFO bank growth + cosine reg (teacher pixels) | Pending |
| E5 | + negative prompts | Nearest competing-class points as SAM negatives | Pending |
| E6 | + confidence fusion | 3-component gate (agreement/boundary/prototype) | Pending |
| E7 | + structural loss | Image-aware boundary smoothness | Pending |
| E8 | Full system | All of the above | Pending |

## Results (full 1,319-image held-out test set)

| Model | mIoU | PA | mPrec / mRecall | Notes |
|---|---|---|---|---|
| Chakravorty et al. (9% dense + points) | 0.6486 | — | — | Reference headline (mixed supervision) |
| **E1 @ ep 5** | **0.4429** | 0.6565 | 0.5909 / 0.6794 | |
| **E1 @ ep 10** | 0.4248 | 0.6532 | 0.5659 / 0.6817 | Non-monotonic dip — decision re-sorting under sparse supervision |
| **E1 @ ep 20** | 0.4758 | 0.6821 | 0.6528 / 0.6410 | |
| **E1 @ ep 30** | 0.4863 | 0.7058 | 0.6649 / 0.6504 | |
| **E1 @ ep 40** | 0.4837 | 0.6985 | 0.6540 / 0.6501 | |
| **E1 @ ep 50** | **0.5035** | 0.7085 | 0.6805 / 0.6708 | Final — ship=0.00 (no annotations); 16-class mean 0.5350 |
| E2 @ … | 🔄 training | | | Same points/seed; eval at ep5/10/20/30/40/50 |

**E1 (v2) details**: ship (class 13) has zero training points in the paper's
annotations → 0.00 IoU; excluding ship, E1 (v2) scores 0.5350 mIoU. Worst
failure: full-frame bare-soil scenes predicted entirely as trees (chapa_1762..
1767, IoU 0.000). For reference, v1 E1 on self-generated points reached
0.4627 → 0.5105 → **0.5188 (ep50)** — but v1 had ship supervised; the v2
ladder compares E1..E8 against each other, and against Chakravorty's headline.
Full analysis: `progress_report_03.md`.

## Commands (from repo root)

```bash
# train
.venv/bin/python -u -m point_only_sam_rs_Es_5pt.run_experiment \
  --config /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/configs/e1_point_only.py \
  --save-dir /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/checkpoints \
  --log /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/runs/logs/E1_train.log

# evaluate
.venv/bin/python -u -m point_only_sam_rs_Es_5pt.run_experiment --evaluate \
  --config /home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/configs/e1_point_only.py \
  --checkpoint /abs/path/point_only_sam_rs_Es_5pt/runs/checkpoints/E1_epoch_0005.pt \
  --save-preds /abs/path/point_only_sam_rs_Es_5pt/runs/predictions/E1_ep5 \
  --log /abs/path/point_only_sam_rs_Es_5pt/runs/logs/E1_ep5_eval.log
```

## House rules

1. **Absolute paths for CLI args** — `resolve()` double-anchors
   repo-root-relative paths into nested dirs. Always pass absolute
   `--save-dir` / `--log` / `--save-preds`.
2. Detach long runs with `setsid bash -c '…' &` (nohup dies with the shell).
3. One checkpoints / logs / predictions dir each — never sibling `runs/` trees.
4. Never rename checkpoints after the fact (docs + eval commands reference them).
5. Eval-time prototype refinement is OFF by default (`proto_use_refine_at_eval`).
   To compare: add `--use-refine`.
