# Paper / implementation gaps

## Comparison basis

Two supplied papers are relevant:

1. Chakravorty & Chakraborty, WACVW 2026, *Point-to-dense supervision
   framework for SAM based remote sensing segmentation with prototype
   refinement and structure aware, confidence guided topology learning*.
   Workspace reports present it as the direct mixed-supervision reference.
2. PointSAM, arXiv:2409.13401v2, *Pointly-Supervised Segment Anything Model
   via Point-To-Mask Distillation*. Workspace notes use it as inspiration for
   LoRA, self-training, prototypes/FIFO, and negative prompts.

Current v2 is not a literal implementation of either paper. It is a semantic
segmentation adaptation borrowing ideas from both. The comparisons below are
against the supplied PDFs, not against unavailable supplementary code.

## Gaps against Chakravorty WACVW 2026

| Paper method/claim | Current v2 implementation | Consequence |
|---|---|---|
| mixed supervision: dense subset plus point-labeled remainder | 630 train / 1,319 held-out test split; trainer receives only point tuples | different split and supervision protocol; not a direct reproduction |
| SAM adaptation on dense labels with `Lseg=alpha*Lsup+beta*Ltopo`; gradual unfreezing | frozen base SAM, LoRA plus new semantic decoder, point CE | removes core dense adaptation and uses a different model/optimization |
| 100 epochs, Adam, batch 4, lr 1e-4 | 50 epochs, AdamW, batch 1, lr 1e-4 | optimizer, effective update count, batch size differ |
| dense class-pixel prototypes from frozen SAM, 16x16 patches, threshold .45 | up to five point features/image, 3x3 pooling, live self/teacher FIFO, EMA .95, cosine acceptance .30 | prototype source, feature region, lifetime, and thresholds differ |
| uncertain-pixel prototype distance refinement | default evaluation has no refinement; training uses cosine regularization; optional refine adds cosine logits | only conceptually related, not paper rule |
| mask-level fusion chooses initial/refined mask using entropy, hierarchical spatial score, boundary score | per-pixel fixed fusion: semantic softmax .45 + hard SAM claim .25 + prototype softmax .30 | no paper mask-level winner, entropy, or 5x5 spatial-consistency method |
| confidence selects whole masks using point agreement plus curvature/contour score and `mu-kappa*sigma`; rare classes monitored | per-pixel gates; E4 uses per-image percentile; E6 uses point/Canny-morphology/prototype score; point CE is class weighted | no selected confident-mask dataset, curvature equation, or paper rare-class selection |
| prompted pseudo mask strictly matches every supplied point | final semantic decoder argmax is not clamped at points; points are CE supervision and a gate score | point agreement encouraged, not guaranteed |
| topology loss compares predicted and dense-GT gradients | E7 is image-edge-weighted probability smoothness | different structural prior; no dense-GT topology loss |
| reported full-model DLRSD mIoU 64.86 | local v2 peaks: E1 .5035, E2 .5409, E3 .5500, E4 .5326 | 64.86 is a non-comparable paper reference, not a reproduction |

### Dense-GT access contradiction

v2 `data/make_manifests.py` reads Chakravorty point masks, but if a class
(documented as ship) has no point while present in dense train GT, it samples
up to five interior dense-GT points. Therefore the loop is point-only, but the
end-to-end data preparation is not strictly free of dense-GT information. The
present v2 manifest has 10,466 points.

The v2 README is stale here: it says 10,310 points and zero ship training
points, contradicting the manifest and its own builder path.

## Gaps against PointSAM

| PointSAM paper | Current v2 | Consequence |
|---|---|---|
| instance segmentation on NWPU VHR-10, HRSID-inshore, WHU; per-instance prompt IoU | 17-class dense semantic segmentation on DLRSD; global-pixel confusion mIoU | task, data, outputs, metric aggregation differ |
| LoRA rank 4, Adam lr .0005, batch 1 | rank 8, AdamW lr .0001, batch 1 | hyperparameters/optimizer differ |
| focal, Dice, IoU mask losses plus Hungarian prototype matching loss | point CE, pseudo CE, KL, cosine regularizers, optional smoothness | all stated PointSAM mask/matching objectives absent |
| FINCH clustering of target/predicted prototypes and Hungarian assignment | one centroid/semantic class and FIFO means/direct cosine | no clustering or Hungarian matching |
| negative prompt calibration uses pairwise predicted-mask IoU to select overlapping positives as negatives | nearest competing semantic-class labelled points, no mask-overlap IoU | v2 negative prompts are an adaptation, not PointSAM NPC |
| teacher mask directly supervises student mask | fused per-pixel teacher distribution from semantic decoder, SAM claims, prototypes | pseudo-label construction differs |

PointSAM supports only the general inspiration claim. It is not experimental
validation for the current semantic segmentation method.

## Implementation/documentation inconsistencies

| Finding | Evidence | Effect |
|---|---|---|
| two E5s share one artifact name | `e5_neg_prompts.py` and `e5_class_aware_gate.py` both set `experiment="E5"` | logs/checkpoints can overwrite and provenance is ambiguous |
| existing E5 checkpoints are class-aware, not negative-prompt | saved checkpoint config has class-aware true and negative prompts false | RUNBOOK/README E5 description mismatches actual artifacts |
| docs call E1--E8 a one-flag chain | E4 changes multiple parameters; E5 has an alternate branch | ladder is not unambiguous |
| v2 README/runs README status is stale | checkoints/logs exist through E5 but README says E2 training and E3+ pending | readers can infer incorrect state |
| evaluation does not restore checkpoint config | evaluator builds from CLI config and uses `strict=False` | mismatched config can silently partially load/alter evaluation |
| no resume state | optimizer/RNG/FIFO omitted | exact continuation is impossible |
| root entrypoint is dense by default | root `Config.dense_ce=True` | accidental root run invalidates point-only claim |
| report script is invalid Python | v2 `make_e4_figs.py:139` indentation error | that report path is non-reproducible |
| plotting dependency undeclared | report scripts import matplotlib but `requirements.txt` omits it | clean environment cannot guarantee report generation |

## Required qualification for any result claim

1. Describe v2 as point-only **training-loop** supervision, unless dense-GT
   ship supplementation is disclosed or removed.
2. Call Chakravorty's 64.86 a reported mixed-supervision reference, not a
   reproduced baseline or directly comparable result.
3. Check checkpoint-embedded config before calling E5 a negative-prompt run.
4. Do not identify v2's fusion/prototype/smoothing equations with either
   paper's exact methods.
5. State that standard evaluation uses the semantic decoder; SAM prompt masks
   and prototype refinement are not part of default inference.

## Uncertain items

- Exact line-by-line relationship between root E0 code and the archived paper
  notebooks: **UNCERTAIN**; no source history/licence maps code to notebook.
- Whether every v2 checkpoint was produced by this exact source snapshot:
  **UNCERTAIN**; embedded configs strongly support v2 provenance but no source
  hashes or manifest hashes are saved.
- Whether archived root/v1 results used precisely the current manifests:
  **UNCERTAIN**; manifests are regenerable in place.
