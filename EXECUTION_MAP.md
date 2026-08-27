# Execution map

## Executable entrypoints

The three independently runnable program families are below. The current one
is v2. CLI help was run successfully for all three runners.

| Family | Training | Evaluation | Preprocessing | Status |
|---|---|---|---|---|
| v2 current | `.venv/bin/python -m point_only_sam_rs_Es_5pt.run_experiment --config <v2 config>` | same command plus `--evaluate --checkpoint <pt>` | `.venv/bin/python -m point_only_sam_rs_Es_5pt.data.make_manifests` | true current path |
| v1 superseded | `.venv/bin/python -m point_only_sam_rs.run_experiment --config <v1 config>` | same plus `--evaluate --checkpoint <pt>` | `.venv/bin/python point_only_sam_rs/data/make_manifests.py`; optional `-m ...data.make_point_images` | historical path |
| root E0 legacy | from repo root: `PYTHONPATH=.. .venv/bin/python -m point_only_semseg.train` | analogous `...point_only_semseg.evaluate <pt>` | `.venv/bin/python make_manifests.py` | dense/mixed default |

Plain root `python train.py`/`python evaluate.py` fail due to relative imports.
`python -m point_only_semseg` invokes root `train(Config())`, whose default
has `dense_ce=True`. `python -m point_only_sam_rs` still requires `--config`.
There is no v2 `__main__.py`; use its explicit `run_experiment` module.

## True v2 training execution path

```text
run_experiment.main
  -> load_config(path) [imports CONFIG dynamically]
  -> train.train(cfg)
     -> PointOnlyDataset(train.json, training=True) + PairAugment
     -> DataLoader(batch=1, shuffle=True)
     -> PointOnlySAM(SAM ViT-B + injected LoRA + semantic decoder)
     -> AdamW(trainable parameters)
     -> optional PrototypeBank initialization across all train items
     -> each batch:
        EMA teacher weak forward -> optional SAM prompt maps -> make_pseudo
        student strong forward -> point CE + enabled terms -> backward/step
        optional teacher EMA and prototype FIFO updates
     -> epochs divisible by save_every: torch.save checkpoint
```

The only dynamic dispatch into the v2 trainer is `run_experiment.py`.
`runs/launch_e2.sh`, `launch_e4.sh`, and `launch_e4_real.sh` are thin wrappers
around it. `auto_pipeline.sh` and `launch_e4_after_e3.sh` are dated,
PID-specific orchestration scripts that can wait and then invoke the same
runner. `consolidate_e3.sh` copies files from another hard-coded workspace; it
does not train.

## True v2 inference/evaluation execution path

```text
run_experiment.main --evaluate
  -> evaluate.evaluate(cfg, checkpoint)
     -> PointOnlyDataset(val.json, training=False)
     -> PointOnlySAM built from current CLI config
     -> torch.load(checkpoint); select teacher if available else student
     -> image -> encode -> semantic_logits -> optional --use-refine -> argmax
     -> global confusion matrix -> mIoU/PA/mPrec/mRecall
     -> optional coloured prediction/GT/point/overlay PNGs
```

Standard inference does not call SAM prompt masks, pseudo-label generation,
teacher updates, or losses. It uses the semantic decoder only. `--use-refine`
is opt-in and off by default.

## v2 imports reached by the entrypoints

| Module | Direct internal imports | External/runtime imports |
|---|---|---|
| `run_experiment.py` | `configs.base`, `train`, `evaluate` | argparse, importlib |
| `train.py` | config; core losses/prompts/prototypes/pseudo; data dataset; model wrapper/lora | torch, numpy |
| `evaluate.py` | config; class map/dataset; model wrapper; lazy colors/prototypes/prompts | torch, cv2, numpy |
| `model/sam_wrapper.py` | decoder, lora | torch, `segment_anything` |
| `model/lora.py` | none | torch |
| `data/dataset.py` | none | cv2, numpy, torch |
| `data/make_manifests.py` | class map | cv2, numpy |
| `core/pseudo.py` | losses, prompts | torch |
| `core/prototypes.py` | none | torch, cv2 |
| `core/losses.py` | none | torch, cv2, numpy |

The config modules use bare imports (`from base import Config`) after the
runner prepends the config directory to `sys.path`. `resolve()` may anchor
new relative outputs below the package directory; absolute CLI paths are the
least ambiguous option.

## Configuration-to-execution mapping

| Config | Effective additions | Artifact evidence |
|---|---|---|
| E1 | point CE | checkpoints/evals 5--50 |
| E2 | live self-updated prototype bank + self cosine regularizer | checkpoints/evals 5--50 |
| E3 | EMA teacher, SAM prompt masks, pseudo CE, KL | checkpoints/evals 5--50 |
| E4 | teacher FIFO/cosine, adaptive percentile gate, cosine warmup | checkpoints/evals 5--50 |
| E5 class-aware | class threshold/prototype-agreement gate | checkpoints 5--30 prove it was used |
| E5 negative prompts | nearest competing-class negative points | source config exists; separate artifact identity unavailable |
| E6/E7/E8 | source configs only | no v2 checkpoints found |
| `_smoke_e3`, `_smoke_e4` | forced short gates/warmup | smoke logs only |

Both E5 config files set `experiment="E5"`, causing identical checkpoint/log
names. The inspected E5 checkpoint config was the class-aware variant, not the
negative-prompt variant.

## Inputs and outputs

| Item | Path/format | Used by |
|---|---|---|
| SAM checkpoint | `sam_vit_b_01ec64.pth` | model construction |
| v2 train data | `point_only_sam_rs_Es_5pt/data/train.json`, 630 items/10,466 points/no masks | trainer |
| v2 eval data | `.../data/val.json`, 1,319 remapped masks/no points | evaluator |
| source point maps | `dlrsd/point_1cmasks/*.png` | manifest builder only |
| dense train masks | `dlrsd/train_1cmasks/*.png` | manifest builder's missing-class supplementation only |
| test masks | `dlrsd/test_for_compare/full_test_1cmasks/*.png` | manifest builder only; remapped outputs are evaluated |
| checkpoints | `runs/checkpoints/E*_epoch_XXXX.pt` | evaluator |
| logs/PNGs | `runs/logs`, `runs/predictions` | reports/visual inspection only |

The evaluator's optional visual point map uses manifest points, or, for test
images without points, samples display-only points from GT. Those display
points never enter model prediction or metrics.

## Checkpoint and metric behavior

Checkpoints store model config, epoch, student, optional teacher, and optional
prototypes. They omit optimizer state, RNG state, FIFO contents, and a resume
cursor. Evaluation constructs the architecture from the supplied config and
uses `load_state_dict(strict=False)`, so a mismatched config can silently alter
what is evaluated.

Evaluation accumulates `M[gt,pred]` globally over every valid test pixel.
`IoU_c=TP/(TP+FN+FP)`, `Precision_c=TP/(TP+FP)`,
`Recall_c=TP/(TP+FN)`; mIoU/mPrec/mRecall are `nanmean` across classes and
`PA=trace(M)/sum(M)`. This is not per-image mean IoU.

## Other executable material

- v1/v2 manifest builders and v1 point-image builder are explicit output
  producers, not automatically invoked by training.
- `runs/collect_results.py`, `runs/analyze_e3_ep35_subset.py`, and figure
  scripts parse existing logs/PNGs. They cannot affect training or official
  metrics.
- Bundled Jupyter notebooks are never imported or shell-invoked by any runner.
  Their execution order/environment is **UNCERTAIN**.
- `point_only_sam_rs_Es_5pt/runs/report_figs/make_e4_figs.py` cannot execute:
  `IndentationError` line 139.

## Verification performed

- v1 runner help, v2 runner help, and root package trainer help: passed.
- v1/v2/root core modules compiled: passed.
- v2 E1--E5 checkpoints were read: E1 student-only; E2 prototypes; E3--E5
  teacher plus prototypes.
- No training/evaluation was started during reconstruction.
