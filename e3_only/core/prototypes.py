"""Class prototypes derived from point annotations — with CORRECT lifetime.

The old codebase's fatal flaw: prototypes were initialised from epoch-0 encoder
features and then frozen for the entire run. As LoRA shifted the feature space,
the prototypes went stale — the distillation targets drifted away from reality
and eval-time refinement with stale prototypes corrupted predictions.

This bank fixes the lifecycle with three cooperating mechanisms:

* ``initialize_from_dataset`` — epoch-0 init from ALL training points.
* ``update_from_predictions``   — EVERY training step, feed the model's own
  high-confidence predictions (softmax max > threshold) into the per-class
  FIFO bank and EMA-update the prototypes. No teacher needed. This is the
  fundamental staleness fix: prototypes track the encoder continuously.
* ``refresh``                   — full re-initialisation from all training
  points with the CURRENT encoder every N epochs (e.g. 10), keeping prototypes
  anchored to real point features rather than drifting on self-predictions.

``refine_consistency`` is GONE. It was a KL self-distillation toward
prototype-refined predictions that (a) barely fired and (b) with stale
prototypes taught the decoder wrong labels. The correct use of live prototypes
is cosine regularisation (``cosine_reg_self``), which aligns the student's
features with the class centroids on confident pixels — no distillation of
unlabelled regions, no way to teach phantom classes.
"""
import time

import torch
import torch.nn.functional as F
from collections import deque


class PrototypeBank:
    def __init__(self, num_classes: int, feat_dim: int, ema: float = 0.95,
                 patch: int = 3, image_size: int = 256, device: str = "cuda",
                 bank_capacity: int = 512, sim_threshold: float = 0.30):
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.ema = ema
        self.patch = patch
        self.image_size = image_size
        self.device = device
        self.sim_threshold = sim_threshold
        self.prototypes = torch.zeros(num_classes, feat_dim, device=device)
        self.initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.refresh_epoch = 0                 # epoch prototypes were last (re)initialised
        # per-class FIFO memory bank of L2-normalized features
        self.bank = {c: deque(maxlen=bank_capacity) for c in range(num_classes)}

    # ------------------------------------------------------------------ #
    #  point feature extraction                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _point_features(self, feat: torch.Tensor, points: torch.Tensor):
        """Patch-pooled, L2-normalized features at each labelled point."""
        b, d, h, w = feat.shape
        k = max(1, self.patch // 2)
        pooled = F.avg_pool2d(feat, kernel_size=self.patch, stride=1, padding=k)
        rows = []
        for x, y, c in points.detach().cpu().tolist():
            if int(c) < 0 or int(c) >= self.num_classes:
                continue
            fx = float(x) * (w / float(self.image_size))
            fy = float(y) * (h / float(self.image_size))
            gx = (fx / max(1.0, (w - 1))) * 2.0 - 1.0
            gy = (fy / max(1.0, (h - 1))) * 2.0 - 1.0
            grid = torch.tensor([[[[gx, gy]]]], device=feat.device, dtype=feat.dtype)
            z = F.grid_sample(pooled, grid, mode="bilinear", align_corners=True)[0, :, 0, 0]
            rows.append((int(c), z))
        return rows

    # ------------------------------------------------------------------ #
    #  whole-training-set (re)initialisation                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def initialize_from_dataset(self, model, items, image_size: int, device: str):
        """Aggregate point features over ALL training images -> one robust
        prototype per class. Used at epoch 0 and by ``refresh``."""
        sums = torch.zeros(self.num_classes, self.feat_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)
        n_imgs = 0
        for item in items:
            image = _load_image(item["image"], image_size, device)
            points = torch.tensor(_scale_points(item.get("points", []), item, image_size),
                                  dtype=torch.float32, device=device)
            if len(points) == 0:
                continue
            feat = model.encode(image)          # one encoder call per image
            for cls, z in self._point_features(feat, points):
                sums[cls] += z
                counts[cls] += 1
            n_imgs += 1
        for c in range(self.num_classes):
            if counts[c] > 0:
                self.prototypes[c] = F.normalize(sums[c] / counts[c], dim=0)
                self.initialized[c] = True
        # seed the FIFO bank with the init features so EMA starts from real data
        for c in range(self.num_classes):
            self.bank[c].clear()
            if counts[c] > 0:
                self.bank[c].append(F.normalize(sums[c] / counts[c], dim=0))
        print(f"[prototypes] initialised {int(counts.gt(0).sum())}/{self.num_classes} "
              f"classes from {int(counts.sum())} labelled points over {n_imgs} images")

    @torch.no_grad()
    def refresh(self, model, items, image_size: int, device: str, epoch: int):
        """Full re-initialisation from ALL training points with the CURRENT
        encoder. Call every ``proto_refresh_every`` epochs so prototypes stay
        anchored to real point features."""
        t0 = time.time()
        self.initialize_from_dataset(model, items, image_size, device)
        self.refresh_epoch = epoch
        print(f"[prototypes] refresh at epoch {epoch} took {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    #  cosine machinery                                                  #
    # ------------------------------------------------------------------ #
    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        """Cosine similarity of every pixel feature to every class prototype."""
        p = F.normalize(self.prototypes, dim=1)
        z = F.normalize(feat, dim=1)
        return torch.einsum("bdhw,cd->bchw", z, p)

    def prototype_confidence(self, feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-pixel prototype evidence: cosine sim of each pixel to the
        prototype of its (predicted) class, mapped to [0, 1]. Uninitialised-class
        pixels get 0.0 (no evidence), not 0.5."""
        b, d, h, w = feat.shape
        z = F.normalize(feat, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        sims = torch.einsum("bdhw,cd->bchw", z, p)
        lab = labels
        if lab.shape[-2:] != (h, w):
            lab = F.interpolate(lab.float().unsqueeze(1), size=(h, w),
                                mode="nearest").long()[:, 0]
        lab = lab.clamp(0, self.num_classes - 1)
        conf = sims.gather(1, lab.unsqueeze(1))[:, 0]
        conf = (conf + 1.0) / 2.0
        uninit = ~self.initialized[lab]
        conf = conf * uninit.logical_not().float()
        return conf

    # ------------------------------------------------------------------ #
    #  LIVE self-supervised updates (the staleness fix)                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_from_predictions(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                                conf_threshold: float = 0.85, sim_threshold: float = 0.30,
                                max_pixels_per_class: int = 256) -> int:
        """Every training step: enqueue the model's OWN high-confidence
        predictions (softmax max > threshold) into the FIFO bank and EMA-update
        the prototypes. No teacher, no pseudo-label machinery. Returns the
        number of pixels accepted (for diagnostics)."""
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        # semantic_logits are at IMAGE resolution (256x256) while feat is at
        # FEATURE resolution (64x64). Align probs to the feature map so the
        # per-pixel indices match feat_flat — otherwise the first confident
        # pixel crashes with an out-of-bounds gather (device-side assert).
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear",
                                  align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= conf_threshold
        if not valid.any():
            return 0
        feat_flat = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)
        labels_flat = labels[0].reshape(-1)
        conf_flat = conf_map[0].reshape(-1)
        valid_flat = valid[0].reshape(-1)
        proto = F.normalize(self.prototypes, dim=1)
        n_accepted = 0
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((labels_flat == cls) & valid_flat &
                              (conf_flat >= conf_threshold))[0]
            if idx.numel() == 0:
                continue
            z = feat_flat[idx]
            sims = (z @ proto[cls]).clamp(-1, 1)
            z = z[sims >= sim_threshold]
            if z.numel() == 0:
                continue
            if z.shape[0] > max_pixels_per_class:
                z = z[torch.randperm(z.shape[0], device=z.device)[:max_pixels_per_class]]
            for vec in z.tolist():
                self.bank[cls].append(torch.tensor(vec, device=self.device))
            if self.bank[cls]:
                bank_mean = F.normalize(torch.stack(list(self.bank[cls])).mean(0), dim=0)
                self.prototypes[cls] = F.normalize(
                    self.ema * self.prototypes[cls] + (1 - self.ema) * bank_mean, dim=0)
                n_accepted += z.shape[0]
        return n_accepted

    @torch.no_grad()
    def update_from_pixels(self, feat: torch.Tensor, labels: torch.Tensor,
                           conf_map: torch.Tensor, valid: torch.Tensor,
                           pixel_threshold: float = 0.80,
                           sim_threshold: float = 0.30,
                           max_pixels_per_class: int = 256) -> int:
        """E4: enqueue gated TEACHER pseudo-label pixels into the FIFO bank.
        Same structure as update_from_predictions but uses teacher features
        and teacher pseudo-labels (not self-predictions).
        Returns the number of pixels accepted."""
        b, d, h, w = feat.shape
        if labels.shape[-2:] != (h, w):
            labels = F.interpolate(labels.float().unsqueeze(1), size=(h, w),
                                   mode="nearest").long()[:, 0]
            conf_map = F.interpolate(conf_map.unsqueeze(1), size=(h, w),
                                     mode="bilinear", align_corners=False)[:, 0]
            valid = F.interpolate(valid.float().unsqueeze(1), size=(h, w),
                                  mode="nearest").bool()[:, 0]
        feat_flat = F.normalize(feat[0].permute(1, 2, 0).reshape(-1, d), dim=1)
        labels_flat = labels[0].reshape(-1)
        conf_flat = conf_map[0].reshape(-1)
        valid_flat = valid[0].reshape(-1)
        proto = F.normalize(self.prototypes, dim=1)
        n_accepted = 0
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((labels_flat == cls) & valid_flat &
                              (conf_flat >= pixel_threshold))[0]
            if idx.numel() == 0:
                continue
            z = feat_flat[idx]
            sims = (z @ proto[cls]).clamp(-1, 1)
            z = z[sims >= sim_threshold]
            if z.numel() == 0:
                continue
            if z.shape[0] > max_pixels_per_class:
                z = z[torch.randperm(z.shape[0], device=z.device)[:max_pixels_per_class]]
            for vec in z.tolist():
                self.bank[cls].append(torch.tensor(vec, device=self.device))
            if self.bank[cls]:
                bank_mean = F.normalize(torch.stack(list(self.bank[cls])).mean(0), dim=0)
                self.prototypes[cls] = F.normalize(
                    self.ema * self.prototypes[cls] + (1 - self.ema) * bank_mean, dim=0)
                n_accepted += z.shape[0]
        return n_accepted

    # ------------------------------------------------------------------ #
    #  cosine regularisation (the E2 supervision, replaces distillation) #
    # ------------------------------------------------------------------ #
    def cosine_reg_self(self, feat: torch.Tensor, semantic_logits: torch.Tensor,
                        threshold: float = 0.85) -> torch.Tensor:
        """Align the mean student feature of confident self-predicted pixels
        with the (detached) class prototype — a pull toward live centroids at
        locations the model itself is sure about. No teacher, no unlabelled
        distillation. Uses softmax-max confidence (not a gate)."""
        b, d, h, w = feat.shape
        probs = torch.softmax(semantic_logits, dim=1)
        # same resolution alignment as update_from_predictions (see above)
        if probs.shape[-2:] != (h, w):
            probs = F.interpolate(probs, size=(h, w), mode="bilinear",
                                  align_corners=False)
        conf_map = probs.max(1).values
        labels = probs.argmax(1)
        valid = conf_map >= threshold
        z = feat[0].permute(1, 2, 0).reshape(-1, d)
        y = labels[0].reshape(-1)
        c = conf_map[0].reshape(-1)
        v = valid[0].reshape(-1)
        losses = []
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((y == cls) & v & (c >= threshold))[0]
            if idx.numel() < 2:
                continue
            zp = F.normalize(z[idx].mean(0), dim=0)
            ref = F.normalize(self.prototypes[cls].detach(), dim=0)
            losses.append(1.0 - torch.sum(zp * ref))
        if not losses:
            return feat.sum() * 0.0
        return torch.stack(losses).mean()

    def cosine_reg(self, feat: torch.Tensor, labels: torch.Tensor,
                    conf_map: torch.Tensor, valid: torch.Tensor,
                    threshold: float) -> torch.Tensor:
        """E4: align the mean student feature of gated TEACHER pseudo pixels
        with the (detached) class prototype. Same idea as cosine_reg_self but
        uses the teacher's gated pseudo-labels instead of self-predictions."""
        b, d, h, w = feat.shape
        if labels.shape[-2:] != (h, w):
            labels = F.interpolate(labels.float().unsqueeze(1), size=(h, w),
                                   mode="nearest").long()[:, 0]
            conf_map = F.interpolate(conf_map.unsqueeze(1), size=(h, w),
                                     mode="bilinear", align_corners=False)[:, 0]
            valid = F.interpolate(valid.float().unsqueeze(1), size=(h, w),
                                  mode="nearest").bool()[:, 0]
        z = feat[0].permute(1, 2, 0).reshape(-1, d)
        y = labels[0].reshape(-1)
        c = conf_map[0].reshape(-1)
        v = valid[0].reshape(-1)
        losses = []
        for cls in range(self.num_classes):
            if not self.initialized[cls]:
                continue
            idx = torch.where((y == cls) & v & (c >= threshold))[0]
            if idx.numel() < 2:
                continue
            zp = F.normalize(z[idx].mean(0), dim=0)
            ref = F.normalize(self.prototypes[cls].detach(), dim=0)
            losses.append(1.0 - torch.sum(zp * ref))
        if not losses:
            return feat.sum() * 0.0
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ #
    #  engagement diagnostics (no more flying blind)                     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def engagement_stats(self, feat: torch.Tensor, labels=None):
        """How much is the prototype mechanism touching? Mean top-1 cosine per
        class and the overall mean top-1 cosine. Logged every 10 steps."""
        if not self.initialized.any():
            return {"per_class_top1": [0.0] * self.num_classes, "pixels_fraction": 0.0}
        sims = self.logits(feat)                          # (1, C, Hf, Wf)
        top1 = sims.max(1).values
        per_class = []
        if labels is not None:
            # labels are at image resolution; sims at feature resolution
            if labels.shape[-2:] != top1.shape[-2:]:
                lab = F.interpolate(labels.float().unsqueeze(1), size=top1.shape[-2:],
                                    mode="nearest").long()[:, 0]
            else:
                lab = labels
            for c in range(self.num_classes):
                m = (lab == c)
                per_class.append(float(top1[m].mean()) if m.any() else float("nan"))
        else:
            per_class = [float(top1.mean())] * self.num_classes
        return {"per_class_top1": per_class, "pixels_fraction": float(top1.mean())}


def _load_image(path: str, image_size: int, device: str) -> torch.Tensor:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0


def _scale_points(points, item, image_size: int):
    import numpy as np
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3).copy()
    if len(pts):
        pts[:, 0] *= image_size / item.get("width", image_size)
        pts[:, 1] *= image_size / item.get("height", image_size)
    return pts
