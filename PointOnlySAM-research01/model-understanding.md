# PointOnlySAM — Model Understanding

> A point-only, dense semantic segmentation model built on top of SAM (Segment Anything Model).
> Trains a semantic decoder using **only sparse point annotations** — no pixel-level masks.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Architecture](#2-architecture)
3. [Code Walkthrough](#3-code-walkthrough)
4. [Training Workflow](#4-training-workflow)
5. [Inference Workflow](#5-inference-workflow)
6. [Data Flow](#6-data-flow)
7. [Loss Functions](#7-loss-functions)
8. [Key Design Decisions](#8-key-design-decisions)

---

## 1. High-Level Overview

### Purpose

PointOnlySAM answers a single research question:

> **Can we train a competitive remote-sensing segmentation model using only a handful
> of point clicks per image, instead of expensive pixel-level mask annotations?**

Traditional semantic segmentation requires annotating every pixel. On the DLRSD
dataset (17 land-cover classes, 256×256 images), a single dense mask takes
minutes to annotate. A handful of point clicks per class takes seconds.

### What the model does

1. **Encodes** each image through SAM's frozen ViT-B image encoder (a powerful
   pretrained feature extractor).
2. **Decodes** those features into per-pixel class predictions via a lightweight CNN head.
3. **Self-trains** using a teacher–student framework: the teacher generates pseudo-labels,
   SAM provides shape priors from point prompts, and a prototype bank provides
   class-level feature statistics — all without ever seeing a dense mask.

### Key capabilities

| Capability | Description |
|---|---|
| **Point-only supervision** | Training uses only `(x, y, class)` annotations — ~5 points per class per image |
| **SAM shape priors** | SAM's mask decoder generates class-agnostic masks from point prompts |
| **Teacher–student self-training** | EMA teacher generates pseudo-labels for unlabeled pixels |
| **Illumination invariance** | Shadow augmentation + consistency loss makes the model robust to lighting |
| **Prototype memory bank** | EMA-updated class prototypes refine predictions via cosine similarity |

---

## 2. Architecture

### High-level architecture diagram

```
                          ┌─────────────────────────────────────────────┐
                          │          PointOnlySAM Architecture          │
                          └─────────────────────────────────────────────┘

  Image (B, 3, 256, 256)
       │
       ▼
  ┌──────────────────────┐
  │   SAM ViT-B Encoder  │  ← Frozen (optionally LoRA-adapted)
  │   (image → 64×64×256)│
  └──────────┬───────────┘
             │
             ├─── weak_features (B, 256, 64, 64)  ──→  [no grad, for teacher/pseudo]
             │
             └─── strong_features (B, 256, 64, 64) ──→  [with grad, for decoder]
                     │
                     ▼
            ┌────────────────────────┐
            │   Semantic Decoder     │  ← Only trainable component
            │   Conv2d(256→192→128→C)│
            └───────────┬────────────┘
                        │
                        ▼
            logits (B, 17, 256, 256)
                        │
            ┌───────────┼───────────┐───────────────────┐
            ▼           ▼           ▼                   ▼
        point_ce    pseudo_ce   shadow_consist    edge_boundary
```

### Component details

#### SAM ViT-B Image Encoder

```
Input:  (B, 3, 1024, 1024)   ← upsampled from 256, scaled ×255, normalized
Output: (B, 256, 64, 64)     ← 16× downsampled feature map
Params: ~89M (all frozen)
```

SAM's ViT-B encoder processes images at 1024×1024 resolution. The output is a
64×64 feature map with 256 channels — rich visual features pretrained on SA-1B
(11M images, 1B masks).

#### Semantic Decoder (the only trainable network)

```python
class SemanticDecoder(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(256, 192, 3, padding=1), nn.GroupNorm(24, 192), nn.GELU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, classes, 1),
        )

    def forward(self, features: torch.Tensor, out_size: int) -> torch.Tensor:
        return F.interpolate(self.head(features), (out_size, out_size),
                             mode="bilinear", align_corners=False)
```

```
Input:  (B, 256, 64, 64)   ← frozen SAM features
Layer1: Conv 256→192 (3×3) + GroupNorm(24) + GELU    → (B, 192, 64, 64)
Layer2: Conv 192→128 (3×3) + GroupNorm(16) + GELU    → (B, 128, 64, 64)
Layer3: Conv 128→128 (3×3) + GroupNorm(16) + GELU    → (B, 128, 64, 64)
Layer4: Conv 128→17  (1×1)                            → (B, 17, 64, 64)
Bilinear upsample                                   → (B, 17, 256, 256)
Output: (B, 17, 256, 256)  ← per-pixel class logits
```

~2.8M parameters. Lightweight by design — the heavy lifting is done by SAM's encoder.

#### SAM Mask Decoder (used for shape priors, not for final prediction)

```python
# For each class, prompt SAM with:
#   - Positive clicks: human-annotated points of that class
#   - Negative clicks: nearest points of other classes (NPC strategy)
low_res, _ = self.sam.mask_decoder(
    image_embeddings=feature.unsqueeze(0),
    image_pe=dense_pe,
    sparse_prompt_embeddings=sparse,
    dense_prompt_embeddings=dense,
    multimask_output=False,
)
```

SAM's mask decoder is called **per class** (17 classes) to generate class-agnostic
shape masks. These are NOT the final predictions — they are geometry priors used
for boundary regularization and pseudo-label confidence.

#### Teacher Decoder (EMA copy)

```python
def make_teacher(decoder: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(decoder)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
```

An exponential moving average (EMA) copy of the decoder. Updated after each step:
```python
def ema_update(student, teacher, decay=0.995):
    for s, t in zip(student.parameters(), teacher.parameters()):
        t.mul_(decay).add_(s, alpha=1.0 - decay)
```

The teacher provides stable pseudo-labels by smoothing out training noise.

#### Prototype Bank (class-level feature memory)

```python
class PointPrototypeBank:
    def __init__(self, classes=17, channels=256, momentum=0.95):
        self.value = torch.zeros(classes, channels)   # (17, 256)
        self.seen  = torch.zeros(classes, dtype=torch.bool)

    def update(self, features, points):
        # For each human-annotated point, EMA-update that class's prototype
        # with the feature vector at that pixel location.
        ...

    def probabilities(self, features):
        # Cosine similarity between every pixel and every prototype → softmax
        scores = einsum("bchw,kc->bkhw", normalize(features), normalize(prototypes))
        return softmax(scores / temperature=0.12), scores.argmax(1)
```

Stores one 256-d feature vector per class. Updated only at human-annotated point
locations (no pseudo-label contamination). Used to provide class-level evidence
for the conservative pseudo-label gate.

---

## 3. Code Walkthrough

### 3.1 Model definition (`pointonlysam/model.py`)

#### LoRA adapter (optional, disabled by default)

```python
class LoRALinear(nn.Module):
    """Low-Rank Adaptation for frozen linear layers."""
    def __init__(self, base, rank, alpha):
        self.base = base                          # frozen original weights
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))   # low-rank A
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))  # low-rank B

    def forward(self, x):
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale
        #     frozen path          trainable low-rank path
```

When `train_sam_lora=True`, LoRA adapters are injected into SAM's attention layers
(qkv and proj) to allow limited fine-tuning of the encoder. Currently **disabled**
(`train_sam_lora: false` in config) — the encoder is fully frozen.

#### The main model class

```python
class PointSAMSemantic(nn.Module):
    sam_side = 1024  # SAM expects 1024×1024 input

    def encode(self, image):
        """Image (B,3,256,256) → features (B,256,64,64)"""
        x = interpolate(image, (1024, 1024))     # upsample to SAM resolution
        x = x * 255.0                             # SAM expects [0, 255] scale
        x = (x - pixel_mean) / pixel_std         # SAM-specific normalization
        with torch.no_grad():                     # frozen encoder = no graph
            return self.sam.image_encoder(x)      # → (B, 256, 64, 64)

    def prompted_geometry(self, feature, points, max_negative=8):
        """Generate per-class SAM masks from point prompts."""
        out = full((17, 256, 256), -10.0)        # default: unclaimed
        for class_id in unique(point_classes):
            positive = points_of_this_class
            negative = nearest_other_points[:max_negative]  # NPC strategy
            sparse, dense = prompt_encoder(coords, labels)
            low_res = mask_decoder(feature, sparse, dense)  # SAM mask decoder
            out[class_id] = low_res
        return out  # (17, 256, 256)
```

### 3.2 Data loading (`pointonlysam/data.py`)

#### Training manifest format

```json
{
    "id": "100",
    "image": "/abs/path/dlrsd/train_images/100.png",
    "width": 256,
    "height": 256,
    "points": [[128, 64, 0], [45, 180, 3], [200, 200, 12], ...]
}
```

Each point is `[x, y, class_id_0_16]`. **No `"mask"` key** — the dataset
constructor raises an error if it finds one (point-only contract enforcement):

```python
if training:
    leaks = [str(x.get("id", i)) for i, x in enumerate(self.items) if "mask" in x]
    if leaks:
        raise ValueError(
            "Point-only contract violated: training manifest contains dense masks"
        )
```

#### Augmentation pipeline (weak + strong views)

```python
class PointPairAugment:
    def __call__(self, image, points):
        # 1. Shared spatial transforms (flip_x, flip_y, rotation)
        #    Points are transformed to stay aligned with pixels
        # 2. Weak view: just normalize + slight brightness jitter
        weak = normalize(image) + uniform(-0.03, 0.03)

        # 3. Strong view: synthetic shadow + contrast + noise
        strong = shadow(image, strength=(0.12, 0.48))   # spatially-varying shadow
        strong = normalize(strong)
        strong = adjust_contrast(strong, (0.75, 1.25))
        strong = add_noise(strong, sigma=0.015)

        return weak, strong, transformed_points
```

Both views share the same spatial transforms so point coordinates remain valid.
The strong view adds photometric variation for the consistency loss.

### 3.3 Training loop (`train.py`)

```python
def main(cfg):
    # Setup
    train_set = PointOnlyDataset(cfg["train_manifest"], cfg["image_size"], training=True)
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"])
    teacher = make_teacher(model.decoder)
    prototypes = PointPrototypeBank(cfg["num_classes"], 256, device, cfg["prototype_momentum"])

    for epoch in range(1, cfg["epochs"] + 1):
        for step, batch in enumerate(loader):
            weak, strong = batch["weak"].to(device), batch["strong"].to(device)
            points = [x.to(device) for x in batch["points"]]

            # ── Phase 1: No-grad targets (teacher, SAM, prototypes) ──
            with torch.no_grad():
                weak_features = model.encode(weak)                    # (B,256,64,64)
                teacher_logits = teacher(weak_features, 256)          # (B,17,256,256)
                proto_prob, _ = prototypes.probabilities(weak_features)
                sam_masks = [model.prompted_geometry(weak_features[i], p)
                             for i, p in enumerate(points)]           # list of (17,256,256)
                labels, confidence, valid = conservative_pseudo(
                    teacher_logits, sam_masks, proto_prob,
                    warm=(epoch <= cfg["warmup_epochs"])
                )

            # ── Phase 2: Decoder forward + loss ──
            strong_features = model.encode(strong)
            logits = model.semantic(strong_features, 256)             # (B,17,256,256)

            loss_point   = point_ce(logits, points)                   # direct point supervision
            loss_pseudo  = weighted_pseudo_ce(logits, labels, confidence, valid)
            loss_shadow  = illumination_consistency(logits, teacher_logits)
            loss_edge    = geometry_boundary_loss(logits, sam_masks, valid)

            loss = (1.0 * loss_point + 1.0 * loss_pseudo +
                    0.25 * loss_shadow + 0.10 * loss_edge)

            loss.backward()
            clip_grad_norm_(params, 1.0)
            optimizer.step()

            # ── Phase 3: EMA update + prototype update ──
            ema_update(model.decoder, teacher, decay=0.995)
            prototypes.update(weak_features, points)
```

### 3.4 Evaluation (`evaluate.py`)

```python
@torch.no_grad()
def main(cfg, checkpoint):
    model = PointSAMSemantic(...)
    state = torch.load(checkpoint)
    model.decoder.load_state_dict(state["teacher_decoder"])  # use EMA teacher

    matrix = zeros(17, 17)  # confusion matrix
    for batch in loader:
        pred = model.semantic(model.encode(image), 256).argmax(1)   # (B, 256, 256)
        # Accumulate confusion matrix
        indices = 17 * target[valid] + pred[valid]
        matrix += bincount(indices).reshape(17, 17)

    # Compute IoU per class
    tp = matrix.diag()
    iou = tp / (matrix.sum(1) + matrix.sum(0) - tp)
    mIoU = iou[iou == iou].mean()  # ignore NaN
```

---

## 4. Training Workflow

### Step-by-step training process

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INITIALIZE                                                   │
│     ├── Load SAM ViT-B (frozen)                                  │
│     ├── Initialize SemanticDecoder (random weights)               │
│     ├── Deep-copy decoder → teacher (EMA)                        │
│     ├── Initialize PrototypeBank (zeros)                         │
│     └── AdamW optimizer on decoder params only                   │
│                                                                  │
│  2. FOR EACH EPOCH (1..30):                                      │
│     │                                                             │
│     │  FOR EACH BATCH:                                           │
│     │                                                             │
│     │  ┌─ A. WEAK VIEW ─────────────────────────────────────┐   │
│     │  │  encode(weak) → weak_features (64×64×256)          │   │
│     │  │  teacher(weak_features) → teacher_logits (256×256×17)│  │
│     │  │  prototypes.probabilities → proto_prob (256×256×17) │   │
│     │  │  prompted_geometry(weak_features, points)            │   │
│     │  │    → 17 SAM masks (256×256 each, one per class)     │   │
│     │  │  conservative_pseudo(...) → labels, conf, valid      │   │
│     │  │                                                      │   │
│     │  │  [All above: no gradients — targets only]           │   │
│     │  └──────────────────────────────────────────────────────┘   │
│     │                                                             │
│     │  ┌─ B. STRONG VIEW ────────────────────────────────────┐   │
│     │  │  encode(strong) → strong_features (64×64×256)       │   │
│     │  │  semantic(strong_features) → logits (256×256×17)    │   │
│     │  │                                                      │   │
│     │  │  Compute 4 losses:                                   │   │
│     │  │    loss_point  = point_ce(logits, points)            │   │
│     │  │    loss_pseudo = weighted_pseudo_ce(...)             │   │
│     │  │    loss_shadow = kl(student ∥ teacher)               │   │
│     │  │    loss_edge   = l1_decoder_edge vs sam_edge        │   │
│     │  │                                                      │   │
│     │  │  loss = Σ weighted_loss_terms                        │   │
│     │  └──────────────────────────────────────────────────────┘   │
│     │                                                             │
│     │  Backward → clip grad → optimizer.step()                   │
│     │  EMA update: teacher += 0.995 * teacher + 0.005 * student  │
│     │  Prototype update: EMA at annotated point locations         │
│     │                                                             │
│  3. SAVE CHECKPOINT                                              │
│     └── {decoder, teacher_decoder, sam_lora, prototypes, config} │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Warmup behavior (epochs 1–3)

During the first `warmup_epochs` (default 3), `conservative_pseudo` sets `valid=False`
for ALL pixels. This means:

- **Epochs 1–3**: Only `point_ce` and `loss_shadow` contribute. No pseudo-labels.
  The model learns from raw point clicks + illumination consistency only.
- **Epoch 4+**: Pseudo-labels activate. The `accepted` rate climbs as the teacher
  becomes more confident and SAM masks become more reliable.

This curriculum prevents noisy pseudo-labels from corrupting the model early in
training when the decoder is still random.

---

## 5. Inference Workflow

```
┌──────────────────────────────────────────────────────────────┐
│                    INFERENCE WORKFLOW                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: RGB image (256, 256, 3)                              │
│                                                              │
│  1. Encode                                                   │
│     image → interpolate to 1024×1024                         │
│     → SAM normalization (×255, subtract mean, divide std)    │
│     → SAM ViT-B encoder (frozen)                             │
│     → features (64, 64, 256)                                │
│                                                              │
│  2. Decode                                                   │
│     features → SemanticDecoder (4-layer CNN)                 │
│     → logits (17, 256, 256)                                 │
│     → bilinear upsample to 256×256                           │
│                                                              │
│  3. Classify                                                 │
│     pred = argmax(logits, dim=0)  → (256, 256)             │
│     Each pixel ∈ {0, 1, ..., 16}                            │
│                                                              │
│  4. Post-processing (optional)                               │
│     - Confusion matrix → per-class IoU                       │
│     - Color map → visualization                              │
│                                                              │
│  Output: (256, 256) integer mask, class ∈ [0, 16]           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

No SAM mask decoding at inference — only the encoder + decoder are used.
SAM's mask decoder is a **training-time tool** for generating geometry priors,
not an inference component.

---

## 6. Data Flow

### Detailed tensor shapes through the pipeline

```
Manifest JSON
│
├── image path → cv2.imread → (256, 256, 3) uint8 RGB
│     │
│     ├── PointPairAugment
│     │     ├── flip/rotate (shared for both views)
│     │     ├── weak:  normalize + jitter  → (256, 256, 3) float32 [0,1]
│     │     └── strong: shadow + contrast + noise → (256, 256, 3) float32 [0,1]
│     │
│     └── permute(2,0,1) + batch → weak (B, 3, 256, 256)
│                                   strong (B, 3, 256, 256)
│
├── points → [(x, y, class), ...] float32
│     └── batch → list of (N_i, 3) tensors (variable length per image)
│
└── (training: no mask key)  (evaluation: mask → (B, 256, 256) int64)


weak (B, 3, 256, 256)
│
├── interpolate → (B, 3, 1024, 1024)
├── × 255 → (B, 3, 1024, 1024)
├── (x - pixel_mean) / pixel_std → (B, 3, 1024, 1024)
└── SAM ViT-B encoder [no_grad]
      └── → weak_features (B, 256, 64, 64)
            │
            ├── teacher.decoder → teacher_logits (B, 17, 64, 64)
            │     └── interpolate → (B, 17, 256, 256)
            │
            ├── prototypes.probabilities → proto_prob (B, 17, 256, 256)
            │     cosine_sim(pixel_features, class_prototypes) / 0.12
            │     softmax → (B, 17, 256, 256)
            │
            └── prompted_geometry (per image, per class):
                  feature[i] (256, 64, 64) + points[i] (N, 3)
                  │
                  ├── for class_id in unique(point_classes):
                  │     positive = points[class == class_id, :2]
                  │     negative = nearest_other_points[:8]
                  │     coords = cat(pos, neg) × (1024/256) → (1, K, 2)
                  │     labels = [1,1,...,1, 0,0,...,0]       → (1, K)
                  │     prompt_encoder(coords, labels) → sparse, dense
                  │     mask_decoder(feature, sparse, dense) → low_res (1, 1, 256, 256)
                  │     out[class_id] = low_res[0, 0]
                  │
                  └── interpolate → sam_masks (17, 256, 256) per image


strong (B, 3, 256, 256)
│
├── interpolate → (B, 3, 1024, 1024)
├── × 255, normalize
├── SAM ViT-B encoder [with grad if LoRA, else no_grad]
│     └── → strong_features (B, 256, 64, 64)
│
└── decoder.head → (B, 17, 64, 64)
      └── interpolate → logits (B, 17, 256, 256)
```

### Memory and computation per step

| Component | Memory | Computation |
|---|---|---|
| SAM ViT-B encode (1024px) | ~1.5 GB VRAM | ~570 ms |
| SAM mask decoder × 17 classes | ~0.3 GB | ~190 ms |
| SemanticDecoder forward | ~0.01 GB | ~1 ms |
| Backward (decoder only) | ~0.1 GB | ~2 ms |
| **Total per step** | **~2.5 GB** | **~760 ms** |

---

## 7. Loss Functions

### 7.1 Point Cross-Entropy (`point_ce`)

```python
def point_ce(logits, point_list):
    """Supervise ONLY at human-annotated point locations."""
    terms = []
    for i, points in enumerate(point_list):
        x = points[:, 0].long()      # pixel x-coords
        y = points[:, 1].long()      # pixel y-coords
        labels = points[:, 2].long()  # class ids
        # Extract logits at (x,y) locations, compute CE
        terms.append(F.cross_entropy(logits[i, :, y, x].t(), labels))
    return mean(terms)
```

**What it does**: Standard cross-entropy, but computed only at the handful of
annotated pixel locations. If 5 classes × 5 points = 25 points, the loss is
computed over just 25 pixels out of 65,536.

**Why it matters**: This is the only "real" supervision signal. Everything else
is self-training.

### 7.2 Weighted Pseudo Cross-Entropy (`weighted_pseudo_ce`)

```python
def weighted_pseudo_ce(logits, labels, conf, valid):
    """Self-training loss on accepted pseudo-labels."""
    if not valid.any():
        return logits.sum() * 0.0   # zero gradient if nothing accepted
    ce = F.cross_entropy(logits, labels, reduction="none")
    return (ce[valid] * conf[valid]).mean()   # weight by confidence
```

**What it does**: Applies cross-entropy at pixels where the pseudo-label gate
(`valid`) is open, weighted by the teacher's confidence.

### 7.3 Conservative Pseudo-Label Gate (`conservative_pseudo`)

```python
def conservative_pseudo(teacher_logits, sam_logits, proto, warm):
    """Only accept pixels where teacher + SAM/prototype agree."""
    tprob = teacher_logits.softmax(1)
    confidence, labels = tprob.max(1)           # teacher's prediction + confidence
    valid = zeros_like(labels, dtype=bool)

    if not warm:  # only after warmup
        for i, masks in enumerate(sam_logits):
            sp = masks.sigmoid()
            sm_conf, sm_label = sp.max(0)       # SAM's prediction per pixel
            sam_agree = (sm_conf >= 0.70) & (sm_label == labels[i])

            proto_agree = zeros_like(sam_agree)
            if proto is not None:
                pc, pl = proto[i].max(0)         # prototype's prediction
                proto_agree = (pc >= 0.55) & (pl == labels[i])

            # Gate: teacher confident AND (SAM agrees OR prototype agrees)
            valid[i] = (confidence[i] >= 0.55) & (sam_agree | proto_agree)

    return labels, confidence, valid
```

**The gate logic** (all three must hold for a pixel to be used):

```
accepted = teacher_confidence ≥ 0.55
           AND
           (
             (SAM confidence ≥ 0.70 AND SAM class == teacher class)
             OR
             (prototype confidence ≥ 0.55 AND prototype class == teacher class)
           )
```

This is the **main defense against confirmation bias**. A pixel is only used
as a training target if two independent sources (SAM geometry or prototype
cosine similarity) agree with the teacher's prediction.

### 7.4 Illumination Consistency (`illumination_consistency`)

```python
def illumination_consistency(student, teacher):
    """KL divergence: student on strong view should match teacher on weak view."""
    return F.kl_div(
        F.log_softmax(student, 1),
        F.softmax(teacher.detach(), 1),
        reduction="none"
    ).sum(1).mean()
```

**What it does**: Forces the decoder's prediction on the shadow-augmented
(strong) image to match the teacher's prediction on the clean (weak) image.
This is a form of data augmentation consistency — the model should produce
the same segmentation regardless of illumination.

### 7.5 Geometry Boundary Loss (`geometry_boundary_loss`)

```python
def geometry_boundary_loss(logits, sam_logits, valid):
    """Align decoder boundaries with SAM's shape priors."""
    probs = logits.softmax(1)
    pred_edge = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean(1)  # decoder edges

    target, support = [], []
    for i, masks in enumerate(sam_logits):
        geometry = masks.sigmoid().max(0).values    # SAM's strongest mask boundary
        edge = (geometry[:, 1:] - geometry[:, :-1]).abs()
        target.append(edge)
        support.append(valid[i, :, 1:] & valid[i, :, :-1])  # only where pseudo-labels exist

    return F.l1_loss(pred_edge[support], target[support])
```

**What it does**: Encourages the decoder's class boundaries to align with SAM's
shape-detected edges. Only enforced at pixels where pseudo-labels are accepted.

### 7.6 Total loss

```python
loss = (
    1.0  * loss_point +     # direct point supervision (always active)
    1.0  * loss_pseudo +    # self-training (active after warmup)
    0.25 * loss_shadow +    # illumination invariance
    0.10 * loss_edge        # boundary regularization
)
```

---

## 8. Key Design Decisions

### Why SAM as backbone?

SAM was trained on 11M images with 1B masks. Its ViT-B encoder produces
extremely rich visual features that generalize well to remote sensing. Using it
as a frozen feature extractor gives us strong features without any pretraining
on DLRSD.

### Why a lightweight decoder?

The decoder is intentionally small (~2.8M params vs SAM's ~89M encoder). This
ensures:
- Fast training (~5.5 min/epoch on 630 images)
- Low VRAM (~2.5 GB)
- The model relies on SAM's features rather than memorizing the training set

### Why weak + strong augmentation (two views)?

The two-view design serves the **illumination consistency loss**:
- **Weak view** → teacher (stable, low augmentation)
- **Strong view** → student (shadow, contrast, noise)

The student must produce the same predictions as the teacher despite the
augmentation. This creates a form of **self-supervised regularization** that
doesn't require any additional labels.

### Why the conservative pseudo-label gate?

Without gating, pseudo-labels create a confirmation bias loop:
1. Model makes wrong prediction → wrong pseudo-label → trains on wrong label
2. Model becomes more confident in wrong prediction → stronger wrong pseudo-label
3. Error amplifies until the model collapses

The gate requires **independent agreement** from SAM or prototypes, breaking
the self-reinforcing loop. If the teacher disagrees with SAM, neither signal
is trusted.

### Why prototypes are updated only at point locations?

If prototypes were updated with pseudo-label pixels, they would inherit the
teacher's biases. By only updating at human-annotated points, prototypes
maintain a clean reference of what each class "looks like" in feature space.
This makes them a reliable second opinion for the pseudo-label gate.

### Why EMA (teacher–student) instead of just training the decoder directly?

EMA smoothing provides:
1. **Stable pseudo-labels** — the teacher changes slowly, so targets don't
   shift dramatically between steps
2. **Noise filtering** — random training noise averages out in the teacher
3. **Better generalization** — EMA models typically泛化 better than the
   fast-changing student

### Why no dense masks in training at all?

The training manifest literally cannot contain a `"mask"` key — the dataset
constructor raises a `ValueError` if one is found. This is a **hard invariant**,
not a soft preference. The entire research question is: *how far can we get
without dense supervision?* Even accidentally using a dense mask would invalidate
the experiment.

### Config summary

| Parameter | Value | Rationale |
|---|---|---|
| `batch_size: 1` | Single image per step | SAM encoding is expensive; gradient accumulation is implicit via full-epoch averaging |
| `lr: 3e-4` | Standard for AdamW | Decayed by EMA, not by scheduler — simple and stable |
| `warmup_epochs: 3` | No pseudo-labels for 3 epochs | Prevents noisy targets early in training |
| `ema_decay: 0.995` | Teacher updates 0.5% per step | Slow enough for stability, fast enough to track improvements |
| `prototype_momentum: 0.95` | Prototypes change 5% per update | Smooth class-level feature estimates |
| `max_negative_points: 8` | NPC strategy | Too many negatives dilute SAM's mask; 8 is a balanced default |
| `w_shadow: 0.25` | Illumination loss weight | Smaller than point/pseudo — regularization, not primary signal |
| `w_boundary: 0.10` | Edge loss weight | Auxiliary — helps but shouldn't dominate |
