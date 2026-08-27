# Research map

## Forensic conclusion

The current research code is `point_only_sam_rs_Es_5pt/` (v2), not the
root-level `train.py` and not `point_only_sam_rs/` (v1). Evidence: all current
launch scripts target v2, its source is newest, and v2 has stored E1--E5
checkpoints. v1 is an earlier nested Git repository; the root package is a
legacy dense/mixed-supervision implementation.

`UNCERTAIN` below marks a conclusion not demonstrable from local imports,
artifacts, source, or supplied PDFs.

## Workspace strata

| Location | Classification | Evidence |
|---|---|---|
| `point_only_sam_rs_Es_5pt/` | current v2 research implementation | active launchers, 46 E1--E5 checkpoints, newest code |
| `point_only_sam_rs/` | superseded v1 | nested Git origin is `github.com/jatinnx/point_only_sam_rs`; archived E1/E2 checkpoints |
| root Python package | legacy E0 dense/mixed baseline | default config enables dense CE and train manifests carry masks |
| `dlrsd_chakraborty_paper/`, `deepglobe-chakraborty-paper/` | isolated prior-work/notebook archive | no v2 import or shell invocation reaches them; authorship/licence **UNCERTAIN** |
| `papers/` | third-party literature | supplied Chakravorty WACVW and PointSAM PDFs |
| prose reports, `MEMORY*.md`, `plan.txt`, `newidea.txt` | history/design notes | not executable authority; may be stale |

`dlrsd` is a symlink to `dlrsd_chakraborty_paper`. It currently provides 630
train images, 630 train dense masks, 630 point masks, 1,319 test images, and
1,319 test masks.

## Origin / third-party assessment

| Material | Assessment |
|---|---|
| `segment_anything` and `sam_vit_b_01ec64.pth` | third-party Meta SAM package/checkpoint, declared in `requirements.txt` |
| both PDFs | third-party source papers |
| v1 package | externally versioned code; nested Git remote above |
| v2 package | workspace-specific derivative/clean rebuild; exact author/licence **UNCERTAIN** |
| root package | local legacy reproduction/derivative; exact origin/licence **UNCERTAIN** |
| paper notebook/archive folders | likely Chakravorty research artifacts, but provenance/licence **UNCERTAIN** |

No workspace-specific source file supplies a licence header.

## v2 component map: FILE -> FUNCTION/CLASS -> CALLED BY -> PURPOSE

| File | Function/class | Called by | Purpose |
|---|---|---|---|
| `run_experiment.py` | `main`, `load_config` | CLI and shell scripts | dynamically loads `CONFIG`; dispatches train/evaluate |
| `configs/base.py` | `Config`, `resolve` | runner/train/evaluate | defaults and path resolution |
| `configs/e1...e8_*.py` | `CONFIG` | `load_config` | experiment variants |
| `train.py` | `train`, `seed_everything`, `build_class_weights` | runner | data/model/optimizer/loss loop; checkpoint save |
| `data/dataset.py` | `PointOnlyDataset`, `PairAugment`, `collate_points` | train/evaluate | loads RGB/masks/points, augments training data, forbids train mask key |
| `model/sam_wrapper.py` | `PointOnlySAM.encode`, `semantic_logits`, `sam_class_logits` | train/evaluate | SAM ViT-B + LoRA semantic model and prompted SAM masks |
| `model/lora.py` | `LoRALinear`, `inject_lora`, `clone_teacher`, `ema_update` | model/train | frozen-base LoRA and EMA teacher |
| `model/decoder.py` | `SemanticDecoder` | model wrapper | SAM feature -> 17 semantic logits |
| `core/losses.py` | point/pseudo CE, KL, smoothness, quality scores | train/pseudo | losses and pseudo-label confidence inputs |
| `core/prototypes.py` | `PrototypeBank` | train/pseudo/eval | point-derived centroids, FIFO updates, cosine evidence/regularization |
| `core/prompts.py` | `NegativePromptSampler` | `sam_class_logits` | class positives plus nearest competing-class negatives |
| `core/pseudo.py` | `sam_hard_labels`, `make_pseudo` | train E3+ | fused teacher labels, confidence, validity map, KL target |
| `evaluate.py` | `evaluate`, `_save_prediction_images` | runner/eval scripts | checkpoint inference, global confusion matrix, outputs |
| `data/make_manifests.py` | `build_train`, `build_val`, `sample_points` | explicit preprocessing only | point JSON and remapped validation masks |
| `data/class_map.py` | `CLASS_NAMES`, `PALETTE` | manifest/evaluate/color code | canonical v2 class order/colours |

## SAM modifications and prompt path

`PointOnlySAM` loads `sam_model_registry["vit_b"]`. `inject_lora` freezes the
SAM image encoder and wraps eligible `qkv`, `proj`, `lin1`, and `lin2` layers
(input width >128) in LoRA. Defaults are rank 8, alpha 16, dropout 0. SAM's
prompt encoder and mask decoder are frozen; the added convolutional
`SemanticDecoder` is trainable.

`encode` bilinearly resizes an RGB 256x256 tensor to 1024x1024 and runs SAM
once. `sam_class_logits` reuses that embedding, scales point coordinates by 4,
and directly calls SAM prompt/mask encoders once per represented class. It does
not use `SamPredictor`. This SAM-mask path is called only while the E3+ teacher
creates pseudo labels, never in standard E1/E2 inference.

## Prompts, masks, and pseudo labels

1. `data/make_manifests.py` reads `dlrsd/point_1cmasks` (values 1--17 class,
   18 unlabelled) and grid-selects up to five points/class. If a class is
   absent there but present in a dense train mask, it adds up to five eroded
   dense-GT points. The supplied v2 manifest has 630 items and 10,466 points.
2. `NegativePromptSampler.sample` groups each semantic class's positive clicks
   and, when enabled, selects at most eight nearest competing-class clicks as
   negatives. `sam_class_logits` turns them into one binary SAM map/class.
3. `make_pseudo` fuses semantic softmax (.45), hard SAM class claims (.25), and
   prototype cosine softmax (.30), normalizes, argmaxes per pixel, and gates
   pixels. E3--E5 confidence is `max(semantic, SAM claim, prototype)`; E6+ is
   `.25*point_agreement + .15*boundary + .60*prototype`. Pseudo labels are
   generated in memory every E3+ batch, not written as a training dataset.

## Losses actually used by v2

| Loss | Implementation | Active experiments |
|---|---|---|
| point CE | CE only at labelled `(x,y,class)` tuples; median-frequency weights capped at 2 | E1--E8, coefficient 1 |
| self prototype cosine | `1-cos(mean(confident feature), detached class centroid)` | E2, coefficient .5, self confidence >=.85 |
| pseudo CE | valid-pixel CE weighted by detached confidence | E3--E8, coefficient warming to 1 |
| consistency | pixelwise `KL(student || fused teacher)` | E3--E8, coefficient 1 |
| teacher prototype cosine | same cosine form, gated teacher pseudo pixels | E4--E8, coefficient .5 |
| boundary smoothness | image-edge-weighted adjacent probability L1 | E7--E8, coefficient .2 |

Quality scores affect gating but are not losses. v2 imports no dense CE, Dice,
focal, IoU, topology, entropy, FINCH, or Hungarian-matching loss. v1's
`refine_consistency` KL and root's dense/topology losses are legacy-only.

## Active data/preprocessing sequence

| Stage | Actual implementation |
|---|---|
| manifest generation | explicit v2 module command; not automatically run by trainer |
| train loading | existing point-only JSON -> RGB/point tensors; mask-key leak is an error |
| train augmentation | shared h/v flips and rotations 0/90/180/270; weak brightness +/- .05; strong brightness +/- .20, contrast [.8,1.2], Gaussian noise .02 |
| model preprocessing | 256 RGB -> 1024 bilinear -> SAM encoder |
| evaluation loading | existing remapped 0..16 validation masks and RGB tensors |

The training loop gets no dense mask. However, the manifest builder's missing
class (ship) supplementation reads dense train masks. Thus “no dense labels in
training” is true only of the loop, not strictly of all data preparation.

## Training configuration that has effect

All fields originate in v2 `configs/base.py`; CLI can override only epochs,
train/validation manifests, save directory/interval, and device.

| Group | Defaults/effect |
|---|---|
| runtime | seed 42; CUDA with CPU fallback; 2 workers |
| data | image 256; 17 classes; no background class |
| SAM | checkpoint `../sam_vit_b_01ec64.pth`; LoRA 8/16/0 |
| optimizer | AdamW, lr 1e-4, wd 1e-4, batch 1, EMA .999, clip 5, 50 epochs |
| class weighting | enabled; inverse median point frequency capped at 2 |
| base coefficients | point 1, pseudo 1, consistency 1, prototype .5, smoothness .2 |
| pseudo schedule | 5-epoch warmup; .70 -> .80/10 epoch threshold ramp; E4+ cosine warmup and adaptive 70th percentile clamp [.30,.80] |
| prototypes | EMA .95; 3x3 point pool; teacher confidence .80; cosine acceptance .30; FIFO 512/class; refresh every 10 epochs; self confidence .85 |
| fusion | semantic/SAM/prototype .45/.25/.30; E6 score .25/.15/.60; SAM claim sigmoid threshold .5 |
| prompts | up to eight negatives |
| flags | prototypes, teacher, SAM masks, prototype regularization, negatives, confidence fusion, smoothness, adaptive/class-aware gates |

E1 is point CE; E2 enables live prototypes; E3 adds teacher and SAM masks; E4
also enables teacher FIFO/cosine, adaptive gate, and cosine warmup; E5 negative
prompts, E6 confidence fusion, E7 smoothness, E8 all of that. A separate
`e5_class_aware_gate.py` is an alternate E5 branch, not a linear successor.

### Complete v2 `Config` field inventory

The table above groups all fields. This exact-name inventory makes the scope
auditable: `experiment`, `seed`, `device`, `num_workers`; `train_manifest`,
`val_manifest`, `image_size`, `num_classes`, `background_class`;
`sam_checkpoint`, `lora_rank`, `lora_alpha`, `lora_dropout`; `epochs`, `lr`,
`weight_decay`, `batch_size`, `ema_decay`, `grad_clip`, `save_dir`,
`save_every`; `use_prototypes`, `use_teacher_student`, `use_sam_prompt_masks`,
`use_proto_reg`, `use_negative_prompts`, `use_confidence_fusion`,
`use_structural_loss`; `l_point`, `l_pseudo`, `l_consistency`, `l_proto`,
`l_smooth`, `pseudo_warmup_epochs`, `pseudo_warmup_cosine`; `pseudo_conf_min`,
`pseudo_conf_max`, `tau_ramp_epochs`, `adaptive_gate`,
`adaptive_gate_percentile`, `adaptive_gate_min`, `adaptive_gate_max`;
`use_class_aware_gate`, `class_pseudo_gate_multi`, `class_agreement_gate`;
`proto_ema`, `proto_feature_patch`, `proto_pixel_confidence`,
`proto_sim_threshold`, `bank_capacity`, `proto_refresh_every`,
`proto_self_conf_threshold`, `proto_use_refine_at_eval`; `fusion_w_sem`,
`fusion_w_sam`, `fusion_w_proto`; `conf_lam_agreement`, `conf_lam_boundary`,
`conf_lam_proto`, `conf_gate_min`, `conf_gate_max`; `max_negative_points`,
`sam_mask_threshold`; `class_weighting`, and `rare_class_factor`.

`proto_use_refine_at_eval` does not alter training, but it does change
evaluation. The default class-aware maps are `{4:1.4, 8:1.3, 12:1.3, 13:1.3}`
and the corresponding default agreement booleans are true; the alternate E5
config replaces them with `{4:1.5, 7:1.3, 11:1.3, 12:1.3}`. The hard-coded
`PairAugment` values and the number of SAM classes/decoder channels also affect
training but are not CLI-configurable.

## Checkpoints and evaluation

Every `save_every` epoch, train saves `<experiment>_epoch_XXXX.pt` with
experiment/epoch/config/student, optional teacher, and allocated prototypes.
It does not save optimizer, RNG, FIFO contents, or resume state. Evaluation
constructs from the *currently supplied config*, loads teacher if present
otherwise student with `strict=False`, and uses prototype refinement only with
`--use-refine`.

For all test pixels, evaluation fills one global 17x17 confusion matrix
`M[ground_truth,prediction]`. For each class, `IoU=TP/(TP+FN+FP)`,
`Precision=TP/(TP+FP)`, and `Recall=TP/(TP+FN)`; mIoU/mPrec/mRecall are
`nanmean` over classes; PA is `trace(M)/sum(M)`. It is global-pixel aggregation,
not mean per-image IoU. It does not calculate F1, Dice, boundary IoU, or
calibration metrics.

Artifact inspection found E1--E4 epochs 5--50 and E5 epochs 5--30. E1 stores
student only; E2 stores 17x256 initialized prototypes; E3--E5 store teacher and
prototypes. The stored E5 configuration is **class-aware gate true, negative
prompts false**.

## Dependencies

Declared: torch >=2.2, torchvision >=.17, NumPy >=1.26, OpenCV >=4.9, Pillow
>=10, and Meta's segment-anything Git package. The present venv has Python
3.12.3, torch 2.13.0+cu130, torchvision .28.0+cu130, NumPy 2.5.2, OpenCV 5.0,
Pillow 12.3, matplotlib 3.11.1, segment_anything, and working CUDA. matplotlib
and PIL are needed by report/visualization scripts but matplotlib is omitted
from `requirements.txt`.

## Dead/duplicated/experimental material

| Material | Classification / reason |
|---|---|
| root package | executable legacy E0; default dense CE; no active runner targets it |
| v1 package | executable superseded implementation; archived E1/E2 only |
| v1/v2 losses, decoder, LoRA, dataset pairs | exact duplicates/renames; each package uses its own copy |
| v1 `e2_diag_norefine.py` | diagnostic-only; no v2 path imports it |
| v2 `_smoke_e3.py`, `_smoke_e4.py` | smoke configs, not normal research ladder |
| v2 class-aware E5 | experimental but artifact-evidenced; checkpoints prove it ran |
| E5 negative/class-aware configs | collision hazard: both use `experiment="E5"` |
| report/analysis scripts | post-processing only; never affect a model or metrics |
| `runs/report_figs/make_e4_figs.py` | report-only and broken: `IndentationError` at line 139 |
| notebooks, old weights/CSVs/PNGs | uncalled archive; exact reproducibility **UNCERTAIN** |
