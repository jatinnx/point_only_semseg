# Shadow Logic — PointOnlySAM

A self-supervised illumination-invariance mechanism that trains the semantic
segmentation model to produce correct predictions regardless of whether
objects lie in sunlight or cast shadow.  The system requires **no shadow
annotations** — it is entirely synthetic and self-supervised.

---

## 1. Synthetic Shadow Injection

**File:** `pointonlysam/fixed_data.py` — `_shadow()` (line 23), `FixedPointPairAugment` (line 37)

During training every image is duplicated into two views:

| View | Description |
|------|-------------|
| **Weak** | Slight brightness jitter (±0.02), no geometric distortion. Fed to the EMA teacher. |
| **Strong** | Same geometric augmentation as weak, plus a randomly generated cast shadow, contrast jitter, and Gaussian noise. Fed to the student. |

### How the synthetic shadow is generated

1. A 16×16 grid of i.i.d. Gaussian samples is drawn.
2. It is Gaussian-blurred (σ ≈ grid/3) to produce smooth, low-frequency variation.
3. The result is bilinearly upsampled to full resolution and min-max normalised to [0, 1].
4. An **attenuation field** is computed as `lo + (hi - lo) × field`, where `lo` and `hi`
   bound the per-pixel darkening (default range: 0.10–0.52, i.e. 10–52 %).
5. Pixels are multiplied by `(1 − attenuation)` to darken them.

The shadow is applied with probability `shadow_probability` (default 0.60 in the
fixed config, 0.75 code-level fallback).

### Shadow mask

The fixed pipeline also emits a **soft shadow-confidence map** used downstream as
a loss weight:

```python
shadow_mask = np.clip((attenuation - 0.08) / 0.32, 0, 1)
```

Only regions where attenuation exceeds 8 % receive a non-zero mask value.  The
resulting map is smooth and continuous, not binary — it reflects *how much* each
pixel is darkened.

---

## 2. Shadow-Invariant RGB Encoder

**File:** `pointonlysam/fixed_model.py` — `ShadowInvariantRGB` (line 55)

Raw RGB is deliberately never fed to the semantic head.  Instead the encoder
derives **8 illumination-resistant channels**:

| Channels | Formula | Rationale |
|----------|---------|-----------|
| 3 chromaticity | `R/(R+G+B)`, `G/(R+G+B)`, `B/(R+G+B)` | Colour ratios are invariant to uniform intensity scaling. |
| 3 local contrast | `pixel / local_mean(15×15)` | Cancels spatially smooth illumination gradients. |
| 2 gradient magnitude | `|Δx(gray)|`, `|Δy(gray)|` | Edges survive shadow transitions; geometry is preserved. |

These are passed through a two-layer CNN stem (Conv → GroupNorm → GELU, repeated)
producing **48 feature channels**.

### Shadow head

A 1×1 convolution on top of those 48 channels outputs a single-channel logit:

```python
shadow_logits = self.shadow_head(f)   # [B, 1, H, W]
```

This predicts, per pixel, whether the input falls inside a shadow region.

---

## 3. Shadow Gating in the Decoder

**File:** `pointonlysam/fixed_model.py` — `FixedSemanticDecoder.forward()` (line 110)

The decoder combines SAM image features with the illumination-normalised
appearance features:

```python
shadow_prob = shadow_logits.sigmoid()
gated_app = app * (1.0 - 0.65 * shadow_prob)     # attenuate up to 65 %
fused = self.fuse(torch.cat((semantic, gated_app), dim=1))
logits = self.class_head(fused)
```

**Effect:** Where the shadow head is confident, the appearance branch is
suppressed and the model leans more on SAM's geometry embeddings — which are
inherently illumination-invariant because SAM was pretrained on diverse lighting
conditions.

During inference `shadow_prob` is **detached**, so the gate operates as a
fixed, non-differentiable routine.

---

## 4. Shadow Disentanglement Loss

**File:** `pointonlysam/fixed_objectives.py` — `shadow_disentanglement_loss()` (line 225)

This is the core training objective.  It enforces **cross-view consistency
inside synthetic-shadow regions**:

```
weight = shadow_mask × (0.25 + 0.75 × teacher_confidence)
loss   = Σ (KL(student ‖ teacher) × weight)  /  Σ weight
```

Step by step:

1. **KL divergence** is computed between the student (shadowed) and teacher
   (clean) log-probabilities at every pixel.
2. The loss is multiplied element-wise by the `shadow_mask`, so only
   synthetic-shadow pixels contribute.
3. An optional **teacher-confidence** soft weight downweights pixels where the
   teacher is uncertain, but never zeroes them out (floor of 0.25).

The intuition: *"Even though this region is darkened, your prediction should
still match what the teacher would produce on the clean image."*

---

## 5. Shadow-Head Supervision

**File:** `pointonlysam/fixed_objectives.py` — `shadow_mask_bce()` (line 272)

A standard binary cross-entropy loss trains the shadow head against the known
synthetic shadow mask:

```python
loss = BCE(shadow_logits.squeeze(1), shadow_mask)
```

This gives the gating mechanism a direct supervision signal, rather than
relying on the disentanglement loss alone to indirectly discover shadow
regions.

---

## 6. Data Flow Diagram

```
Training Image
    │
    ├──── Weak view (clean) ──────────→ Teacher Encoder ──→ Teacher Logits (EMA)
    │
    └──── Strong view (+ shadow) ─────→ Student Encoder
              │
              ├─→ SAM ViT-B (frozen) ──────────────→ Semantic Features
              │                                         │
              └─→ ShadowInvariantRGB ─→ Shadow Head     │
                    │                    │              │
                    │              shadow_prob          │
                    │                    │              │
                    └─→ Gated Appearance ◄──────────────┤
                                         │              │
                                    Concat + Fuse ◄─────┘
                                         │
                                   Student Logits
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
              Shadow Disent.       Point CE /            Shadow Head
                Loss (KL)         Region CE               BCE Loss
                    │                    │                    │
                    └────────────────────┴────────────────────┘
                                         │
                                   Total Loss
```

### Loss weights (`configs/dlrsd_pointonly_sam_fixed.json`)

| Weight | Value | Purpose |
|--------|-------|---------|
| `w_shadow` | 0.40 | KL consistency between student and teacher in shadow regions |
| `w_shadow_head` | 0.25 | BCE supervision for the shadow-detection head |

---

## 7. Why This Matters

In remote-sensing imagery (satellite, aerial, drone), cast shadows from
buildings, clouds, and terrain are pervasive.  Without shadow invariance a
point-supervised model will:

- **Misclassify** shadowed vegetation as water or asphalt (dark = wrong class).
- **Fragment boundaries** at shadow edges where illumination changes abruptly.
- **Learn spurious correlations** between absolute brightness and semantic labels.

The shadow logic solves this without any shadow annotations: the synthetic
augmentation creates paired clean/shadowed examples, and the disentanglement
loss + gated appearance branch together teach the decoder to ignore illumination
and focus on geometry and colour ratios.

---

## 8. Evolution Across Versions

| Aspect | v1 | v2 | Fixed |
|--------|----|----|-------|
| Shadow augmentation | None | `_shadow()` in `data.py` — used for consistency, no mask returned | `_shadow()` in `fixed_data.py` — returns `(image, attenuation)` |
| Shadow mask | N/A | Not tracked | Soft mask from thresholded attenuation; fed to loss and shadow head |
| Shadow head | N/A | None | 1×1 conv in `ShadowInvariantRGB`; trained with BCE |
| Appearance input | Raw RGB | Raw RGB | Illumination-normalised (chromaticity + local contrast + gradients) |
| Gating | None | None | `app × (1 − 0.65 × shadow_prob)` suppresses appearance in shadow |
| Disentanglement loss | `illumination_consistency()` — simple KL on full image | Same | `shadow_disentanglement_loss()` — KL weighted by shadow mask + teacher confidence |
| Config keys | `w_shadow` only | `w_shadow` only | `w_shadow`, `w_shadow_head`, `shadow_probability`, `shadow_teacher_confidence` |


we are not gonna use any dense dataset for making points, like there wont be dense
  dataset point. i have to make a modal to get a perfect point, piture this we have an
  airplane okay it has 3 parts like 1 body and 2 wing sam gonna treat it like 2 different
  part rights??? this could happen to other classes to,  so like we have to make a modal
  that dont do this.