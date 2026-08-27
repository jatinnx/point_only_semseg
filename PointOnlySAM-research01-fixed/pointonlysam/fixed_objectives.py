"""Objectives for the corrected point-only semantic model."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class PrototypeScores:
    scores: torch.Tensor  # [B, C, H, W]
    probabilities: torch.Tensor  # [B, C, H, W]


class MultiPrototypeBank:
    """K prototypes per class, updated only from human point features.

    Each incoming labeled point is assigned to the nearest prototype of its
    class. This preserves multiple modes within a semantic class instead of
    collapsing all appearances to one vector.
    """
    def __init__(self, classes: int, channels: int, device: torch.device,
                 prototypes_per_class: int = 4, momentum: float = 0.95,
                 temperature: float = 0.15):
        self.classes = classes
        self.channels = channels
        self.k = prototypes_per_class
        self.momentum = momentum
        self.temperature = temperature
        self.value = torch.zeros(classes, self.k, channels, device=device)
        self.seen = torch.zeros(classes, self.k, dtype=torch.bool, device=device)

    @torch.no_grad()
    def update(self, features: torch.Tensor, points: list[torch.Tensor], image_size: int) -> None:
        _, _, h, w = features.shape
        for bi, sample in enumerate(points):
            for x, y, c in sample.tolist():
                c = int(c)
                if c < 0 or c >= self.classes:
                    continue
                xx = min(w - 1, max(0, int(x * w / image_size)))
                yy = min(h - 1, max(0, int(y * h / image_size)))
                feat = F.normalize(features[bi, :, yy, xx], dim=0)
                if not self.seen[c].any():
                    slot = 0
                elif not self.seen[c].all():
                    slot = int((~self.seen[c]).nonzero(as_tuple=False)[0].item())
                else:
                    sims = torch.mv(F.normalize(self.value[c], dim=1), feat)
                    slot = int(sims.argmax().item())
                if not self.seen[c, slot]:
                    self.value[c, slot] = feat
                    self.seen[c, slot] = True
                else:
                    self.value[c, slot] = F.normalize(
                        self.momentum * self.value[c, slot] + (1.0 - self.momentum) * feat, dim=0
                    )

    def scores(self, features: torch.Tensor) -> PrototypeScores | None:
        if not self.seen.any():
            return None
        x = F.normalize(features, dim=1)
        p = F.normalize(self.value, dim=2)
        sim = torch.einsum("bdhw,ckd->bckhw", x, p)
        sim = sim.masked_fill(~self.seen[None, :, :, None, None], -1e4)
        scores = sim.max(dim=2).values / self.temperature
        prob = scores.softmax(dim=1)
        return PrototypeScores(scores=scores, probabilities=prob)

    def point_margin_loss(self, features: torch.Tensor, points: list[torch.Tensor], image_size: int,
                          margin: float = 0.15) -> torch.Tensor:
        terms = []
        _, _, h, w = features.shape
        p = F.normalize(self.value, dim=2)
        for bi, sample in enumerate(points):
            for x, y, c in sample.tolist():
                c = int(c)
                if c < 0 or c >= self.classes or not self.seen[c].any():
                    continue
                xx = min(w - 1, max(0, int(x * w / image_size)))
                yy = min(h - 1, max(0, int(y * h / image_size)))
                feat = F.normalize(features[bi, :, yy, xx], dim=0)
                sim = torch.einsum("ckd,d->ck", p, feat).masked_fill(~self.seen, -1e4).max(1).values
                pos = sim[c]
                neg = sim.masked_fill(torch.arange(self.classes, device=sim.device) == c, -1e4).max()
                terms.append(F.relu(margin - (pos - neg)))
        return torch.stack(terms).mean() if terms else features.sum() * 0.0

    def state_dict(self) -> dict[str, torch.Tensor | float | int]:
        return {
            "value": self.value.cpu(), "seen": self.seen.cpu(),
            "momentum": self.momentum, "temperature": self.temperature,
            "k": self.k,
        }

    def load_state_dict(self, state: dict) -> None:
        self.value.copy_(state["value"].to(self.value.device))
        self.seen.copy_(state["seen"].to(self.seen.device))


def point_ce(logits: torch.Tensor, point_list: list[torch.Tensor], class_weights: torch.Tensor | None = None) -> torch.Tensor:
    terms = []
    for i, points in enumerate(point_list):
        if len(points) == 0:
            continue
        x = points[:, 0].long().clamp(0, logits.shape[-1] - 1)
        y = points[:, 1].long().clamp(0, logits.shape[-2] - 1)
        labels = points[:, 2].long()
        terms.append(F.cross_entropy(logits[i, :, y, x].t(), labels, weight=class_weights))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def presence_targets(point_list: list[torch.Tensor], classes: int, device: torch.device) -> torch.Tensor:
    target = torch.zeros(len(point_list), classes, device=device)
    for i, points in enumerate(point_list):
        if len(points):
            ids = points[:, 2].long().unique().clamp(0, classes - 1)
            target[i, ids] = 1.0
    return target


def partial_presence_bce(
    logits: torch.Tensor,
    points: list[torch.Tensor],
    classes: int,
    negative_weight: float = 0.0,
) -> torch.Tensor:
    """Positive-only image-class presence supervision.

    A point annotation proves that its class is present.
    Missing point annotations are treated as unknown, not negative.
    """
    target = presence_targets(points, classes, logits.device)
    known = target > 0.5
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[known],
        target[known],
    )


def force_present_classes(
    presence_prob: torch.Tensor,
    points: list[torch.Tensor],
    classes: int,
) -> torch.Tensor:
    """Force human-point-observed classes to be active during training."""
    out = presence_prob.clone()
    for i, sample in enumerate(points):
        if len(sample) == 0:
            continue
        ids = sample[:, 2].long().unique().clamp(0, classes - 1)
        out[i, ids] = 1.0
    return out


def gate_logits(
    logits: torch.Tensor,
    presence_logits: torch.Tensor,
    threshold: float = 0.20,
    scale: float = 3.0,
    top_k: int = 8,
    forced_presence: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft class-set control.

    Classes below threshold are strongly downweighted, but the top-k classes
    are always retained to preserve useful generalization when the presence
    estimator is uncertain.
    """
    prob = presence_logits.sigmoid()
    if forced_presence is not None:
        prob = torch.maximum(prob, forced_presence)
    active = prob >= threshold
    if top_k > 0:
        k = min(top_k, prob.shape[1])
        top_idx = prob.topk(k, dim=1).indices
        active.scatter_(1, top_idx, True)
    gate = torch.where(active, torch.ones_like(prob), torch.exp(-scale * (threshold - prob).clamp_min(0)))
    gated = logits + torch.log(gate.clamp_min(1e-5)).unsqueeze(-1).unsqueeze(-1)
    return gated, prob


def conservative_pseudo(teacher_logits: torch.Tensor,
                        presence_prob: torch.Tensor,
                        sam_logits: list[torch.Tensor],
                        proto_prob: torch.Tensor | None,
                        warm: bool,
                        class_threshold: float = 0.20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tprob = teacher_logits.softmax(1)
    confidence, labels = tprob.max(1)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    if warm:
        return labels, confidence, valid
    for i, masks in enumerate(sam_logits):
        sp = masks.sigmoid()
        sm_conf, sm_label = sp.max(0)
        sam_agree = (sm_conf >= 0.55) & (sm_label == labels[i])
        proto_agree = torch.zeros_like(sam_agree)
        if proto_prob is not None:
            pc, pl = proto_prob[i].max(0)
            proto_agree = (pc >= 0.45) & (pl == labels[i])
        class_ok = presence_prob[i, labels[i]] >= class_threshold
        valid[i] = (confidence[i] >= 0.50) & class_ok & (sam_agree | proto_agree)
    return labels, confidence, valid


def weighted_pseudo_ce(logits: torch.Tensor, labels: torch.Tensor, conf: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, labels, reduction="none")
    return (ce[valid] * conf.detach()[valid]).mean()


def region_ce(logits: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor,
              class_weights: torch.Tensor | None = None) -> torch.Tensor:
    valid = labels != 255
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=255, reduction="none")
    return (loss[valid] * confidence[valid].detach()).mean()


def shadow_disentanglement_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    shadow_mask: torch.Tensor,
    valid_teacher: torch.Tensor | None = None,
    teacher_confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shadow/illumination consistency loss.

    The loss is applied on synthetic-shadow pixels. If teacher confidence
    is provided, it is used as a soft weighting signal. A hard teacher-valid
    mask may optionally be supplied for backward compatibility.
    """

    shadow = shadow_mask.float()

    # Nothing to optimize if this batch has no shadow pixels.
    if shadow.sum() <= 0:
        return student_logits.sum() * 0.0

    student_logp = F.log_softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits.detach(), dim=1)

    kl = F.kl_div(
        student_logp,
        teacher_prob,
        reduction="none",
    ).sum(dim=1)

    # Start with all synthetic-shadow pixels.
    weight = shadow

    # Optional hard validity mask. Only use it if one is explicitly passed.
    if valid_teacher is not None:
        weight = weight * valid_teacher.float()

    # Soft confidence weighting. Low-confidence teacher predictions are
    # downweighted rather than completely discarded.
    if teacher_confidence is not None:
        conf = teacher_confidence.detach().float().clamp(0.0, 1.0)
        weight = weight * (0.25 + 0.75 * conf)

    denom = weight.sum().clamp_min(1e-6)

    return (kl * weight).sum() / denom

def shadow_mask_bce(shadow_logits: torch.Tensor, shadow_mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(shadow_logits.squeeze(1), shadow_mask.float())


def soft_edge_map(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """2-D semantic discontinuity from horizontal and vertical neighbor agreement."""
    px = (probs[..., 1:] * probs[..., :-1]).sum(1)
    py = (probs[:, :, 1:, :] * probs[:, :, :-1, :]).sum(1)
    return 1.0 - px, 1.0 - py


def boundary_loss_2d(logits: torch.Tensor,
                     boundary_target: torch.Tensor,
                     boundary_support: torch.Tensor,
                     smooth_support: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """True 2-D boundary loss + anti-fragmentation neighborhood smoothness.

    The boundary calculation is intentionally performed in FP32 because
    torch.nn.functional.binary_cross_entropy is not AMP/autocast safe.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits_fp32 = logits.float()
        probs = logits_fp32.softmax(1)
        ex, ey = soft_edge_map(probs)

        tx = boundary_target[..., 1:].float()
        ty = boundary_target[:, 1:, :].float()
        sx = boundary_support[..., 1:].bool()
        sy = boundary_support[:, 1:, :].bool()

        if not (sx.any() or sy.any()):
            zero = logits_fp32.sum() * 0.0
            return zero, zero

        edge_terms = []
        if sx.any():
            edge_terms.append(
                F.binary_cross_entropy(
                    ex[sx].clamp(1e-5, 1.0 - 1e-5),
                    tx[sx],
                )
            )
        if sy.any():
            edge_terms.append(
                F.binary_cross_entropy(
                    ey[sy].clamp(1e-5, 1.0 - 1e-5),
                    ty[sy],
                )
            )
        edge = torch.stack(edge_terms).mean()

        if smooth_support is None:
            smooth = logits_fp32.sum() * 0.0
        else:
            ssx = smooth_support[..., 1:].bool()
            ssy = smooth_support[:, 1:, :].bool()
            smooth_terms = []
            if ssx.any():
                smooth_terms.append(ex[ssx].mean())
            if ssy.any():
                smooth_terms.append(ey[ssy].mean())
            smooth = (
                torch.stack(smooth_terms).mean()
                if smooth_terms
                else logits_fp32.sum() * 0.0
            )

        return edge, smooth
