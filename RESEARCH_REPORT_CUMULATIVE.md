# Point-Only Semantic Segmentation — Cumulative Research Report
**Dataset**: DLRSD (17 classes, aerial remote sensing)  
**Task**: Semantic segmentation using only point annotations (no dense masks at training time)  
**Target**: ≥ 65.0 mIoU (Chakraborty et al. mixed-supervision baseline)  
**Report date**: August 2026  
**Codebase**: `point_only_sam_rs_Es_5pt/` (E-series), `PointOnlySAM-research01/`, `PointOnlySAM-research01-fixed/`, `PointOnlySAM-research02/`

---

## Table of Contents
1. [Dataset and task context](#1-dataset-and-task-context)
2. [Experiment E1 — Point CE baseline](#2-experiment-e1--point-ce-baseline)
3. [Experiment E2 — Prototype bank](#3-experiment-e2--prototype-bank)
4. [Experiment E3 — EMA teacher-student](#4-experiment-e3--ema-teacher-student)
5. [Experiment E4 — Proto regularisation + adaptive gate](#5-experiment-e4--proto-regularisation--adaptive-gate)
6. [research01 v1 — New codebase, first attempt](#6-research01-v1--new-codebase-first-attempt)
7. [research01 v2 — Geometry cache addition](#7-research01-v2--geometry-cache-addition)
8. [research01-fixed — Four targeted fixes](#8-research01-fixed--four-targeted-fixes)
9. [research02 — Point-reasoning architecture](#9-research02--point-reasoning-architecture)
10. [Full issue register](#10-full-issue-register)
11. [Gap analysis: where we are vs 65 mIoU](#11-gap-analysis-where-we-are-vs-65-miou)
12. [What needs to change](#12-what-needs-to-change)

---

## 1. Dataset and task context

**DLRSD** contains 2100 aerial images at 256×256 pixels, 17 semantic classes. The train/val split used throughout is 630 training / 1319 validation images.

**Point annotation contract**: each training image has ~5–17 annotated pixels per class present. No dense pixel masks are used at training time. Validation uses the official dense masks exclusively for metric computation.

**Class distribution is heavily imbalanced**:

| Group | Classes | Total train points |
|---|---|---|
| Dominant | grass, pavement, trees, cars, buildings, bare_soil | 1040–1885 pts each |
| Mid | sand, water | 305–335 pts |
| Rare | airplane, chaparral, court, dock, field, mobile_home, sea, ship, tanks | 130–170 pts each |

**The 65 mIoU target** comes from Chakraborty et al., who reached 64.86 using 9% of images with dense masks plus point annotations — a mixed-supervision setup. All our experiments are point-only.

**Backbone**: SAM ViT-B (segment_anything, vit_b variant). Feature map: 64×64×256 from a 1024×1024 resized input.

---

## 2. Experiment E1 — Point CE baseline

**Codebase**: `point_only_sam_rs_Es_5pt/`  
**Config**: `configs/e1_point_only.py`  
**Epochs**: 50  

### What we did
Simplest possible setup: freeze SAM encoder, add a small 4-layer CNN decoder (256→256→128→17), train with cross-entropy loss only at the annotated point locations. LoRA rank 8 injected into all SAM attention layers (qkv, proj, lin1, lin2).

### Results

| Epoch | mIoU | PA |
|---|---|---|
| ep5 | 0.4429 | 0.6565 |
| ep30 | 0.4863 | 0.7058 |
| ep50 | **0.5035** | 0.7085 |

**Per-class at ep50 (notable failures)**:
- ship: **0.000** — zero annotated training points for ship class
- field: 0.056 — severe under-representation
- bare_soil: 0.288 — spectral confusion with grass and chaparral

### What worked
- LoRA adaptation of the frozen encoder is essential. Without it the domain gap between SA-1B (natural photos) and DLRSD (aerial) makes features useless for discriminating aerial classes.
- Point CE alone gets you to 0.50 — a reasonable baseline given only sparse supervision.
- TTA-4 (averaging predictions over 4 rotations) gives +4.1 mIoU on a 40-image subset and +1.8 mIoU on the full test set, confirming SAM's ViT has orientation sensitivity.

### What went wrong
- Ship: no annotations → zero IoU. The model never sees this class during training.
- Dense regions with no points (most of the image) receive no direct gradient. The decoder generalizes from the point locations but has no signal in ambiguous regions.
- Non-monotonic trajectory: ep10 dips to 0.425 before recovering, typical of sparse supervision instability in early epochs.

---

## 3. Experiment E2 — Prototype bank

**Config**: `configs/e2_prototypes.py`  
**Epochs**: 50 (evaluated at 5/10/20/30/40/50)  

### What we did
Added a live prototype bank on top of E1. Before epoch 1: initialize prototypes by sweeping all 630 training images with the current encoder and computing patch-pooled L2-normalized features at every annotated point location. During training: every step, feed high-confidence self-predictions (softmax max > 0.85) into a FIFO bank (512 entries/class) and EMA-update the prototypes. Every 10 epochs: full refresh from all training points with the current encoder.

This bank then contributes to two things: (1) as a third evidence source in pseudo-label fusion, and (2) as a cosine regularization loss pulling student features toward class centroids.

### Results

| Epoch | mIoU | PA |
|---|---|---|
| ep5 | 0.4429 | — |
| ep30 | **0.5409** | — |
| ep50 | 0.5127 | — |

**Best E2: 0.5409 at ep30 (+3.7 mIoU vs E1@50)**

Notable: ship recovered from 0.00 to 0.63 IoU — the prototype bank gave ship some representation even with few annotated points by pulling in high-confidence self-predictions.

### What worked
- Prototype initialization from the full training set before epoch 1 — gives every class a reasonable centroid immediately.
- FIFO bank + EMA update keeps prototypes tracking the evolving LoRA-adapted encoder. This is the key fix over "initialize and freeze" approaches.
- 10-epoch refresh re-anchors prototypes to real point features, preventing drift.

### What went wrong
- Early stopping at ep30 needed — by ep50 the model decays. Dense self-prediction updates eventually contaminate the bank with overconfident wrong predictions.
- Chaparral/field/bare_soil still confused spectrally. The prototype bank helps for classes that are visually distinct; it can't compensate for the encoder's inability to separate aerially-similar textures.
- Field and rare classes improved marginally only.

---

## 4. Experiment E3 — EMA teacher-student

**Config**: `configs/e3_teacher_student.py`  
**Epochs**: 50  

### What we did
Full teacher-student semi-supervised learning on top of E2. Teacher = EMA copy of the student decoder (decay 0.999). At each step the teacher generates pseudo-labels on the weak-augmented image; a confidence gate decides which pixels become training targets for the student on the strong-augmented image.

**Pseudo-label fusion**: combines three sources with learnable weights:
- `w_sem=0.45`: teacher semantic softmax
- `w_sam=0.25`: hard argmax of class-specific SAM prompt masks (built per-image from point annotations using SAM's mask decoder)
- `w_proto=0.30`: prototype cosine similarity softmax

**Gate**: `max(sem_conf, SAM_claim, proto_conf) ≥ threshold`. Initially a fixed ramp from 0.70→0.80 over 10 epochs. **Bug discovered during E3**: teacher softmax peaks at ~0.67, so a 0.70 threshold means the gate is permanently open — nothing is filtered. Every pseudo-label passes regardless of confidence.

### Results

| Epoch | mIoU | PA |
|---|---|---|
| ep10 | 0.5377 | 0.7250 |
| ep20 | 0.5427 | — |
| **ep30** | **0.5500** | **0.7371** |
| ep40 | 0.5442 | — |
| ep50 | 0.5401 | 0.7341 |

**Best E3: 0.5500 at ep30 (+4.6 mIoU vs E1@50)**. This is the highest mIoU achieved in the entire E-series.

**Per-class at E3@30**:
- chaparral: 0.673 (strong)
- court: 0.806 (strong)
- ship: 0.637 (recovered)
- field: 0.088 (still failing)
- bare_soil: 0.376 (still weak)

### What worked
- Teacher-student pseudo-labeling provides dense supervision everywhere in the image, not just at point locations. This is the main gain over E2.
- Multi-source fusion (semantic + SAM geometry + prototype) is more robust than any single source.
- SAM geometry provides spatial structure that pure semantic confidence cannot.

### What went wrong
- **Gate was permanently open** (the 0.70 threshold was unreachable with the teacher's actual softmax distribution). The model was training on every pixel's pseudo-label regardless of confidence.
- **Chaparral collapses by ep30 in E4** (and to some degree late in E3): teacher mistakes chaparral for bare_soil confidently → pseudo-labels propagate the error → student learns the mistake → EMA teacher becomes more confident about the mistake. Self-reinforcing loop.
- **Hallucination**: sparse scenes get ghost classes. A scene with 4 GT classes gets 9 predicted classes because the open gate lets every moderately confident wrong prediction through.
- Peak at ep30 then degradation: the pseudo-label contamination accumulates over time.

---

## 5. Experiment E4 — Proto regularisation + adaptive gate

**Config**: `configs/e4_proto_reg.py`  
**Epochs**: 50  

### What we did
E4 = E3 + two targeted fixes for the problems observed in E3:
1. **Adaptive percentile gate**: instead of a fixed threshold, compute the 70th-percentile confidence across all pixels in the image and use that as the gate threshold. This guarantees exactly 30% of pixels are gated out per image regardless of the absolute softmax scale. Range clamped to [0.30, 0.80].
2. **Proto cosine regularisation** (`cosine_reg`): a loss term that pulls student features at gated teacher pseudo-label locations toward the detached class prototype centroid. Weight `l_proto=0.5`.

### Results

| Epoch | mIoU | PA |
|---|---|---|
| ep5 | 0.4858 | 0.6820 |
| ep10 | 0.5273 | 0.7176 |
| ep20 | 0.5302 | 0.7269 |
| ep30 | 0.5206 | 0.7247 |
| ep50 | — | — |

**Best E4: 0.5302 at ep20 (−0.020 vs E3@30)**. E4 never beats E3.

**Per-class comparison E4@10 vs E3@50**:
- chaparral: E4=0.442 at ep10, **collapses to 0.227 by ep20, 0.200 by ep30**
- dock: −0.058 vs E3
- ship: −0.041 by ep30

### What worked
- Adaptive gate fixed the "permanently open" problem. The gate now actually filters.
- `cosine_reg` helped some structural classes (buildings, court, tanks) marginally.

### What went wrong
- **Chaparral collapse worsened**: adaptive gate at 70th percentile with `min=0.30` lets very low-confidence pixels through on hard images (gate threshold can be as low as 0.30). Combined with `cosine_reg` pulling features toward wrong centroids (because teacher already misclassifies chaparral as bare_soil), the loop becomes even more destructive than in E3.
- **`cosine_reg` amplifies teacher errors**: it pulls features toward the prototype — but if the pseudo-label for a pixel is wrong, the cosine pull moves that pixel's feature toward the wrong class's centroid. This is the opposite of what was intended.
- Training log shows `bank_px=0` throughout — the `update_from_pixels` teacher bank feed never accepted any pixels. The bank improvement was essentially inactive.

---

## 6. research01 v1 — New codebase, first attempt

**Repo**: `PointOnlySAM-research01/`  
**Config**: `configs/dlrsd_pointonly_sam.json`  
**Epochs**: 30  

### What was new
Completely rewritten codebase from scratch. New components:
- `SemanticDecoder`: 4-layer CNN on SAM features + **RGB fusion branch** (32-channel ConvNet on raw pixels, fused with SAM features via Conv(160→128→17))
- `PointPrototypeBank`: EMA updated only at annotated point locations (not from self-predictions)
- `conservative_pseudo` gate: strict AND-gate requiring `teacher_conf ≥ 0.55 AND (SAM_conf ≥ 0.70 AND agrees OR proto_conf ≥ 0.55 AND agrees)`
- `illumination_consistency`: KL divergence between student (strong-aug) and teacher (clean) — shadow invariance

**LoRA disabled** (`train_sam_lora: false`).

### Results
mIoU at epoch 30 (from eval): **~0.473** (v1 README figure).  
`accepted` pseudo rate: peaks at **~49%** — highest of any experiment.

### What worked
- Strict AND-gate produces cleaner pseudo-labels when it fires.
- Higher pseudo coverage (~49%) than v2 despite the strict gate, because the geometry cache wasn't conflicting with pseudo-labels yet.

### What went wrong
- **LoRA off** → frozen SAM encoder → same spectral confusion as if LoRA never existed. All the new architecture can't compensate for a frozen domain-mismatched encoder.
- **RGB branch introduces salt-and-pepper noise**: 4.73% pixel transition rate vs 2.90% in E3@30. Raw pixel values at 256×256 are too noisy for semantic classification — individual pixel brightness fluctuations get labelled as different classes.
- **Shadow labelling bug**: `illumination_consistency` KL loss trains the student to match the teacher — but the teacher is also confused by shadows (it wasn't trained to know shadows). Both predict shadow pixels as different material; the KL loss reinforces this.
- **Prototype bank too sparse**: updated only at ~25 point locations per image, no full-dataset initialization, no refresh. Bank produces weak prototype guidance.
- Shadow class hallucination visible in output images: airplane shadows predicted as grass/sea.

---

## 7. research01 v2 — Geometry cache addition

**Config**: `configs/dlrsd_pointonly_sam_v2.json`  
**Epochs**: 40  

### What was new
Added a pre-computed **geometry cache**: before training, run SAM once on all 630 training images with the point-annotated classes as prompts, store the resulting binary mask regions. During training, these static labels are used as a `region_ce` loss (`w_region=1.0`).

Also: `w_pseudo` cut from 1.0 to 0.5, `warmup_epochs` reduced from 3 to 2.

### Results
mIoU at epoch 40 (final): **0.4433** — **worse than v1 (0.4726)**.

**Hallucination analysis across 300 images**:
- chaparral hallucinated in 56% of images
- grass: 48%, water: 48%, court: 46%, sea: 32%
- Single class dominating >70% of image: **71/300 images (24%)**
- bare_soil: dominant wrong prediction in 37 images

### What worked
Nothing — v2 is strictly worse than v1.

### What went wrong
- **Geometry cache conflicted with live pseudo-labels**: the static SAM regions were built with a frozen encoder. By epoch 10 the decoder has learned something, but the cache labels reflect the untrained encoder's geometry. `w_region=1.0` gives equal weight to stale static labels as to the live point annotations. The two compete, degrading both.
- **Accepted pseudo rate dropped from 49% to 31%**: the cache's conflicting gradients disrupted the pseudo-label pathway.
- **Augmentation disabled**: the cache was built at a fixed orientation, so flip/rotation augmentation had to be removed to keep cache coordinates valid. Less augmentation → lower generalization.
- **Earlier warmup end (epoch 2 vs 3)**: pseudo-labels start firing before the decoder is stable, creating early-epoch confirmation bias that's hard to undo.
- **Pixel-level noise worst in this version**: 4.73% transition rate, dominant-class collapse in 24% of images.

---

## 8. research01-fixed — Four targeted fixes

**Repo**: `PointOnlySAM-research01-fixed/`  
**Config**: `configs/dlrsd_pointonly_sam_fixed.json`  
**Epochs**: 40  

### What was fixed (from FIXED_MODEL.md)
1. **Class-set hallucination**: added image-level presence head (`partial_presence_bce`) + soft gate on class logits
2. **Prototype collapse**: `MultiPrototypeBank` with 4 prototypes per class, margin loss between positive and nearest negative class
3. **Boundary fragmentation**: 2D boundary loss (both x and y directions) + interior smoothness term
4. **Shadow confusion**: `ShadowInvariantRGB` branch using chromaticity + local-contrast + gradient (not raw RGB), plus shadow-gated consistency loss only on synthetic shadow pixels

**LoRA enabled** (`train_sam_lora: true`, rank 4, start_layer 8).

### Results

| Checkpoint | Mode | mIoU | PA |
|---|---|---|---|
| ep5 | image_only | 0.4154 | 0.6470 |
| ep40 | **point_set** | **0.4561** | **0.6762** |

**Confusion analysis (ep40 point_set, full 1319 images)**:
- chaparral → bare_soil: **71.0%** of GT chaparral pixels mislabelled
- field → grass: **55.8%** of GT field pixels mislabelled
- sea → pavement: 17.6%
- sand → bare_soil (21.3%) + pavement (20.9%)

### What worked
- LoRA re-enabled gave structural improvement
- `ShadowInvariantRGB` reduces shadow noise vs raw RGB branch
- 2D boundary loss reduced spatial noise to 3.74% (down from 4.73%)
- `MultiPrototypeBank` with margin loss is better than single-centroid EMA for classes with multiple visual appearances

### What went wrong
- **Presence head collapsed to loss=0 by epoch 29**: `partial_presence_bce` uses `negative_weight=0.0` so it only supervises present classes. Model learns to output high logits for everything — loss hits zero, gate becomes useless. Hallucination problem not solved.
- **Chaparral still collapses worse than v2** (IoU 0.400 vs v2's 0.608): `cosine_reg` still pulls chaparral features toward the bare_soil prototype because teacher pseudo-labels are still wrong, and the margin loss (0.15) is too small to overcome spectral similarity.
- **No image_only eval for ep40** — the reported 0.4561 is point_set mode (uses test-time point labels, not a blind evaluation). ep5 image_only is 0.4154, which is actually below v2's 0.4433. Real improvement in blind setting is unconfirmed.
- **Pseudo coverage still stuck at 25%**: geometry cache quality is still the binding constraint on pseudo coverage.
- **Field near-zero** (IoU=0.077): only 8 validation images contain field, and the model cannot learn from 130 total training point annotations for a class this underrepresented.
- **Sea regressed vs v2** (−0.142 mIoU): shadow-invariant branch removes absolute brightness cue that distinguishes sea (dark, reflective) from pavement (mid-grey, matte). By normalizing intensity the branch made these two spectrally closer.

---

## 9. research02 — Point-reasoning architecture

**Repo**: `PointOnlySAM-research02/`  
**Config**: `configs/base.json` / `runs/pointreasoning_v1/config.json`  
**Status**: **NEVER TRAINED** — model has zero checkpoints, zero eval images, zero metrics

### What was designed
Major architectural redesign. Core idea: point evidence should be in the semantic forward path, not applied as a post-filter.

**New components**:
- `PointConditioner`: extracts L2-normalized feature anchors at every annotated point, computes dense cosine similarity between every pixel and each class's anchor set → per-class affinity map added directly to base logits
- `SemanticContext`: class-token cross-attention + edge-aware 4-neighbor propagation + learnable C×C class compatibility matrix
- `ShadowInvariantBranch`: same illumination-normalized approach as fixed model, shadow-gated
- Prompt geometry used directly as logit additive (pre-computed SAM masks)

**Two inference modes**:
- image_only: base logits + context + presence prior
- point_conditioned: + point_affinity + prompt_geometry + hard class-set suppression

### Blocking bugs that prevent training
1. **`presence_threshold` missing from both config files** → `KeyError` on training step 1 → model never runs
2. **`train_sam_lora: false`** → frozen encoder, same spectral blindness as research01 v1
3. **No flip/rotation augmentation** → prompt cache was built at fixed orientation, geometric augmentation removed → weaker invariance than E3

### Code-level assessment of each general issue

| Issue | research02 handling | Verdict |
|---|---|---|
| Spectral ambiguity | PointConditioner helps in point-conditioned mode only | Same problem at image-only inference |
| Pseudo coverage ceiling | 4-condition AND-gate (teacher≥0.80 AND geometry≥0.65 AND agree AND conf≥0.10) | **Worse** — estimated ~15-22% coverage |
| Rare class failure | Uniform WeightedRandomSampler weights | Unchanged |
| Prototype bank | Replaced by affinity loss at point locations | Better for annotated regions, no help elsewhere |
| Confirmation bias | Geometry cross-check reduces but doesn't eliminate | Partially improved |
| Boundary fragmentation | SemanticContext edge-aware propagation + compat matrix | **Improved** — strongest fix in research02 |
| Train/test distribution gap | 70% of training steps have point_affinity+geometry; image-only inference doesn't | **New regression** introduced |

### Realistic ceiling if bugs were fixed

| Scenario | Estimated mIoU |
|---|---|
| Fixed crash + LoRA off | 0.43–0.51 (mode-dependent) |
| Fixed crash + LoRA rank 8 | 0.52–0.56 (mode-dependent) |

---

## 10. Full issue register

Issues in roughly causal order — upstream problems cause downstream ones.

### Structural (cannot be fixed by adding loss terms)

**S1 — Encoder domain gap**  
SAM ViT-B was trained on SA-1B (natural ground-level photos). DLRSD is top-down aerial. Features produced by the frozen encoder cannot separate chaparral from bare_soil, field from grass, sand from pavement because these distinctions don't exist in SA-1B.  
*Present in*: research01 v1/v2 (LoRA off), research02 (LoRA off)  
*Partially mitigated in*: E1–E4, research01-fixed (LoRA rank 8 or rank 4)

**S2 — LoRA rank insufficient**  
Even with LoRA on, rank 4 from layer 8 of 12 only adapts the last 4 attention blocks. Rank 8 across all 12 blocks gives substantially better domain adaptation. No experiment used rank ≥ 16 or fine-tuned layer norms.  
*Present in*: research01-fixed, research02-if-enabled

**S3 — Sparse supervision budget**  
~5–17 annotated pixels per image. 8 rare classes have ≤155 total training points across the entire 630-image dataset. This is the root constraint from which most other issues flow.  
*Present in*: all experiments (inherent to the task)

**S4 — Pseudo-label coverage ceiling**  
All gate designs converge to 15–49% accepted coverage. The gate is structurally limited by the teacher's confidence, which is itself limited by sparse point training. Adding more gate conditions makes quality marginally better but coverage much worse.

| Experiment | Accepted % |
|---|---|
| E3 (open gate) | ~100% of images, quality poor |
| research01 v1 | ~49% |
| research01 v2 | ~31% |
| research01-fixed | ~25% |
| research02 (projected) | ~15–22% |
| E3/E4 reference | ~49% |

### Training dynamics

**T1 — Teacher-student confirmation bias**  
Wrong teacher prediction → passes gate → student trains on it → EMA teacher becomes more confident about same wrong prediction → next epoch more pseudo-labels with the same error. Visible as the chaparral collapse: IoU 0.44@ep10 → 0.22@ep20 → 0.20@ep30.  
*Present in*: E3, E4, research01 variants, research02

**T2 — Presence head trivially solved with zero signal**  
`partial_presence_bce` with `negative_weight=0.0` only supervises annotated-positive classes. Model outputs high logits for all classes → loss→0 → gate useless → hallucination continues.  
*Present in*: research01-fixed, research02

**T3 — Cosine regularisation amplifies teacher errors**  
`cosine_reg` (E4) and `point_margin_loss` (research01-fixed) pull student features toward the class prototype. If the teacher's pseudo-label for a pixel is wrong (e.g. chaparral→bare_soil), the cosine pull moves that pixel's feature toward the wrong centroid. The loss actively worsens the confusion it was meant to prevent.  
*Present in*: E4, research01-fixed

**T4 — Prototype bank data starvation**  
E2/E3/E4 bank: ~5 points × ~630 images = ~3150 updates for dominant classes, ~130–170 for rare classes. Single-centroid EMA of 130 samples cannot represent the visual variance within a class. Fixed model's 4-prototype-per-class bank is better but still sparse.  
*Present in*: E2, E3, E4, research01 variants

**T5 — Class imbalance in sampling not sufficiently compensated**  
Pavement has 1885 annotated points, field has 130 — 14.5× imbalance. Inverse-sqrt weighting reduces this to ~3.8× effective imbalance. Rare classes still receive far fewer gradient updates.  
*Present in*: all experiments

### Architecture / inference

**A1 — RGB fusion branch introduces pixel-level noise**  
Adding a raw-pixel CNN branch means adjacent pixels with slightly different brightness get different class predictions. Produces salt-and-pepper segmentation maps. 4.73% horizontal transition rate in research01 v2, vs 2.90% in E3@30.  
*Present in*: research01 v1/v2

**A2 — Shadow mislabels pixels as different class**  
Shadow darkens pavement → features shift toward bare_soil/water in embedding space. The illumination consistency loss (KL between student-with-shadow and teacher-without-shadow) doesn't fix this because the teacher is also confused by the same shadow. Both sides of the KL are wrong.  
*Present in*: research01 v1/v2 (via KL), research01-fixed (partially mitigated by `ShadowInvariantRGB`)

**A3 — Image-only inference is out-of-distribution (research02)**  
70% of research02 training steps use point_affinity + prompt_geometry in the logit computation. At image-only inference both are absent. The decoder learned to rely on signals it won't see at test time.  
*New in*: research02

### Configuration / code bugs

**B1 — `presence_threshold` missing from config (research02)**  
`KeyError` on first training step. Model never runs.

**B2 — `train_sam_lora: false` in research01 v1/v2 and research02**  
The most impactful single bug. Costs ~3–4 mIoU by leaving the encoder frozen in a mismatched domain.

**B3 — Geometry cache vs live pseudo-label conflict (research01 v2)**  
Static SAM regions built before training conflict with evolving pseudo-labels after epoch 10. `w_region=1.0` gives the stale cache equal weight as point annotations. v2 is worse than v1 specifically because of this.

**B4 — Augmentation disabled for cache alignment (research01 v2, research02)**  
To keep prompt cache coordinates valid, flip/rotation augmentation was removed. Model trains without geometric invariance. TTA-4 experiment on E1 showed +4.1 mIoU on a 40-image subset from rotation averaging — the encoder is orientation-sensitive and needs rotation training to compensate.

**B5 — E3 gate permanently open**  
Teacher softmax peaks at ~0.67. Fixed threshold of 0.70 is never reached. Every pseudo-label pixel passes the gate regardless of confidence quality. This is why E3 shows hallucination (ghost classes in sparse scenes) while also having the highest mIoU — it's an accidental lucky balance where the pseudo-labels happen to be mostly right despite being unfiltered.

---

## 11. Gap analysis: where we are vs 65 mIoU

### Results table (all experiments)

| Experiment | mIoU | PA | Eval mode | LoRA |
|---|---|---|---|---|
| E1@50 | 0.5035 | 0.7085 | standard | rank 8 |
| E1@50 + TTA-4 | 0.5215 | 0.7227 | TTA | rank 8 |
| E2@30 | 0.5409 | — | standard | rank 8 |
| **E3@30** | **0.5500** | **0.7371** | standard | rank 8 |
| E3@30 + TTA-4 (projected) | ~0.570 | — | TTA | rank 8 |
| E4@20 | 0.5302 | 0.7269 | standard | rank 8 |
| research01 v1@30 | 0.4726 | — | standard | off |
| research01 v2@40 | 0.4433 | 0.6320 | standard | off |
| research01-fixed@5 | 0.4154 | 0.6470 | image_only | rank 4 |
| research01-fixed@40 | 0.4561 | 0.6762 | **point_set*** | rank 4 |
| research02 | **N/A** | **N/A** | not trained | off |
| **Chakraborty et al.** | **0.6486** | — | — | — |
| **Target** | **0.650** | | | |

*point_set uses test-time point labels — not a blind evaluation

### Per-class comparison: best of each system vs E3@30

| Class | E1@50 | E3@30 | research01 v2 | r01-fixed@40ps | Gap to 65 (from E3) |
|---|---|---|---|---|---|
| airplane | 0.543 | 0.569 | 0.464 | 0.552 | — |
| bare_soil | 0.288 | 0.376 | 0.336 | 0.364 | needs +0.37 |
| buildings | 0.560 | 0.602 | 0.368 | 0.499 | needs +0.11 |
| cars | 0.599 | 0.626 | 0.419 | 0.509 | needs +0.10 |
| chaparral | 0.488 | **0.673** | 0.608 | 0.400 | needs +0.04 |
| court | 0.790 | **0.806** | 0.613 | 0.573 | ok at E3 |
| dock | 0.418 | 0.448 | 0.305 | 0.384 | needs +0.25 |
| field | 0.056 | 0.088 | 0.267 | 0.078 | needs +0.56 |
| grass | 0.557 | 0.570 | 0.463 | 0.510 | needs +0.08 |
| mobile_home | 0.411 | 0.448 | 0.322 | 0.456 | needs +0.22 |
| pavement | 0.724 | 0.730 | 0.596 | 0.660 | ok |
| sand | 0.361 | 0.390 | 0.356 | 0.301 | needs +0.27 |
| sea | 0.668 | 0.681 | **0.681** | 0.539 | ok at E3/v2 |
| ship | 0.000 | 0.637 | 0.464 | 0.429 | needs +0.10 |
| tanks | 0.636 | 0.665 | 0.397 | 0.516 | needs +0.05 |
| trees | 0.511 | 0.530 | 0.400 | 0.439 | needs +0.10 |
| water | 0.718 | 0.728 | 0.477 | 0.545 | needs +0.06 |
| **mIoU** | **0.503** | **0.550** | **0.443** | **0.456** | **need +0.10** |

**The 10 mIoU gap from E3@30 to 65 is not uniform.** Field alone needs +0.56 IoU improvement — it's effectively unsolved. Bare_soil needs +0.37. Dock, sand, mobile_home each need +0.22–0.27. If these four classes were solved to competitive level the overall mIoU would reach ~0.62–0.64.

---

## 12. What needs to change

The three changes below are necessary conditions for reaching 65. Each addresses a structural bottleneck that no current experiment has solved.

### Change 1 — Stronger encoder adaptation

Current best: LoRA rank 8 on all 12 attention blocks (E1–E4). This gave +3–4 mIoU over frozen encoder. But rank 8 still leaves the encoder partially domain-mismatched — chaparral and bare_soil features remain too close in embedding space.

Needed:
- **LoRA rank ≥ 16** on all attention layers, or
- **Layer norm fine-tuning** in all ViT blocks (cheap: adds only 2 × hidden_dim parameters per block, ~50K total, but highly effective for domain shift because layer norms control the activation distribution), or
- Both

This directly addresses Issue S1 and S2, and indirectly reduces the spectral confusion in Issues S3/T1.

### Change 2 — Per-image class centroid pseudo-label gate

All current gates depend on the SAM geometry sigmoid being ≥0.65 as a condition. SAM geometry is class-agnostic — it finds segments, not classes. For spectrally ambiguous classes (chaparral, field, bare_soil) the geometry sigmoid is unreliable because SAM doesn't know what "chaparral" looks like.

Replace the SAM-geometry dependency in the pseudo gate with:

1. At each training step, compute the mean L2-normalized feature of all same-class annotated point pixels in the batch → one per-image class centroid per present class
2. Accept pseudo-labels wherever cosine similarity to the nearest class centroid exceeds a threshold (e.g. 0.5)
3. Reject pseudo-labels where the nearest centroid is a different class than the teacher's prediction (breaks confirmation bias)

This approach accepts pixels based on feature-space proximity to ground-truth points rather than teacher confidence, which is fundamentally more reliable for spectrally similar classes.

### Change 3 — Dense SAM supervision for rare classes

8 classes have ≤155 training point annotations total. The pseudo-label gate almost never fires for these classes because the teacher is rarely confident enough about them. They're stuck in a catch-22: too few points to train well, too little training to pass the gate, too little gate-passing to improve.

Use the pre-computed SAM prompt masks (which already exist in the geometry cache) differently:
- For every class present in an image (as indicated by point annotations), use the full SAM binary mask for that class as a soft CE target, weighted by the mask confidence score
- Apply this as a separate loss term (`w_sam_dense`) alongside point CE, not as a gate condition
- Weight by inverse class frequency so rare classes get proportionally more gradient from this term

This provides dense supervision for rare classes at every training step regardless of teacher confidence, bypassing the catch-22.

---

*Report covers experiments E1–E4 (August 17–19 2026) and research01 v1/v2/fixed, research02 (August 17–25 2026). All results on DLRSD 1319-image held-out validation set.*
