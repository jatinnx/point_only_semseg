"""Point, pseudo-label, illumination, geometry, and prototype objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


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


class PointPrototypeBank:
    """EMA semantic prototypes updated only at labelled human point locations."""
    def __init__(self, classes: int, channels: int, device: torch.device, momentum: float = 0.95):
        self.value = torch.zeros(classes, channels, device=device)
        self.seen = torch.zeros(classes, dtype=torch.bool, device=device)
        self.momentum = momentum

    @torch.no_grad()
    def update(self, features: torch.Tensor, points: list[torch.Tensor]) -> None:
        h, w = features.shape[-2:]
        for bi, sample in enumerate(points):
            for x, y, c in sample.tolist():
                xx = min(w - 1, max(0, int(x * w / 256)))
                yy = min(h - 1, max(0, int(y * h / 256)))
                c = int(c)
                feat = F.normalize(features[bi, :, yy, xx], dim=0)
                self.value[c] = feat if not self.seen[c] else self.momentum * self.value[c] + (1 - self.momentum) * feat
                self.value[c] = F.normalize(self.value[c], dim=0)
                self.seen[c] = True

    def probabilities(self, features: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.seen.any():
            return None, None
        x = F.normalize(features, dim=1)
        p = F.normalize(self.value, dim=1)
        scores = torch.einsum("bchw,kc->bkhw", x, p) / 0.12
        scores[:, ~self.seen] = -1e4
        prob = torch.softmax(scores, dim=1)
        return prob, scores.argmax(1)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {"value": self.value.cpu(), "seen": self.seen.cpu(), "momentum": self.momentum}

    def load_state_dict(self, state: dict) -> None:
        self.value.copy_(state["value"].to(self.value.device))
        self.seen.copy_(state["seen"].to(self.seen.device))


def conservative_pseudo(teacher_logits: torch.Tensor, sam_logits: list[torch.Tensor],
                        proto: torch.Tensor | None, warm: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accept a pixel only when semantic prediction agrees with SAM or PBR.

    This deliberately produces partial labels; ignored pixels never create a
    self-training target.  It is the main protection against pseudo-label
    confirmation bias.
    """
    tprob = teacher_logits.softmax(1)
    confidence, labels = tprob.max(1)
    valid = torch.zeros_like(labels, dtype=torch.bool)
    if not warm:
        for i, masks in enumerate(sam_logits):
            sp = masks.sigmoid()
            sm_conf, sm_label = sp.max(0)
            sam_agree = (sm_conf >= 0.70) & (sm_label == labels[i])
            proto_agree = torch.zeros_like(sam_agree)
            if proto is not None:
                pc, pl = proto[i].max(0)
                proto_agree = (pc >= 0.55) & (pl == labels[i])
            valid[i] = (confidence[i] >= 0.55) & (sam_agree | proto_agree)
    return labels, confidence, valid


def weighted_pseudo_ce(logits: torch.Tensor, labels: torch.Tensor, conf: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not valid.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, labels, reduction="none")
    return (ce[valid] * conf.detach()[valid]).mean()


def region_ce(logits: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor,
              class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Cross entropy on static point-seeded SAM interiors; 255 is ignored."""
    valid = labels != 255
    if not valid.any():
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=255, reduction="none")
    return (loss[valid] * confidence[valid].detach()).mean()


def illumination_consistency(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(student, 1), F.softmax(teacher.detach(), 1), reduction="none").sum(1).mean()


def geometry_boundary_loss(logits: torch.Tensor, sam_logits: list[torch.Tensor], valid: torch.Tensor) -> torch.Tensor:
    """Align boundaries only on high-confidence, point-seeded SAM regions."""
    probs = logits.softmax(1)
    pred = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean(1)
    target, support = [], []
    for i, masks in enumerate(sam_logits):
        geometry = masks.sigmoid().max(0).values
        edge = (geometry[:, 1:] - geometry[:, :-1]).abs()
        target.append(edge)
        support.append(valid[i, :, 1:] & valid[i, :, :-1])
    target_t, support_t = torch.stack(target), torch.stack(support)
    if not support_t.any():
        return logits.sum() * 0.0
    return F.l1_loss(pred[support_t], target_t[support_t])
