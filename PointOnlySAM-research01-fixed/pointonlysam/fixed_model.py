"""Fixed PointOnlySAM model.

Point-only semantic segmentation with four explicit safeguards:
1. image-level class-presence gating (no free 17-way hallucination at inference),
2. multi-prototype semantic guidance for intra-class variation,
3. shadow-gated high-resolution appearance features,
4. a 2-D boundary-aware semantic decoder.

No dense target masks are used by the model or its training code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / max(rank, 1)
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


def install_lora(image_encoder: nn.Module, rank: int, alpha: float, start_layer: int) -> None:
    for p in image_encoder.parameters():
        p.requires_grad = False
    for index, block in enumerate(image_encoder.blocks):
        if index < start_layer:
            continue
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)


@dataclass
class DecoderOutput:
    logits: torch.Tensor
    presence_logits: torch.Tensor
    shadow_logits: torch.Tensor
    embedding: torch.Tensor


class ShadowInvariantRGB(nn.Module):
    """High-resolution branch based on illumination-normalized cues.

    Raw RGB is deliberately not fed directly to the semantic head. We use
    chromaticity, local-mean normalized color, and 2-D gradients so darkening
    from shadows has less leverage while edges remain available.
    """
    def __init__(self, channels: int = 48):
        super().__init__()
        self.local = nn.AvgPool2d(15, stride=1, padding=7)
        # 3 chromatic + 3 local contrast + 2 gradient magnitude channels.
        self.stem = nn.Sequential(
            nn.Conv2d(8, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, channels, 3, padding=1), nn.GroupNorm(8, channels), nn.GELU(),
        )
        self.shadow_head = nn.Conv2d(channels, 1, 1)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-4
        mean = image.mean(1, keepdim=True)
        chroma = image / (image.sum(1, keepdim=True) + eps)
        local_mean = self.local(mean).clamp_min(0.04)
        local_norm = image / local_mean
        gray = mean
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        gx = F.pad(dx.abs(), (0, 1, 0, 0))
        gy = F.pad(dy.abs(), (0, 0, 0, 1))
        features = torch.cat((chroma, local_norm.clamp(0, 4), gx, gy), 1)
        f = self.stem(features)
        shadow_logits = self.shadow_head(f)
        return f, shadow_logits


class FixedSemanticDecoder(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.classes = classes
        self.sam_head = nn.Sequential(
            nn.Conv2d(256, 192, 3, padding=1), nn.GroupNorm(24, 192), nn.GELU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
        )
        self.appearance = ShadowInvariantRGB(48)
        self.fuse = nn.Sequential(
            nn.Conv2d(176, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
        )
        self.class_head = nn.Conv2d(128, classes, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, classes),
        )
        self.embedding_head = nn.Sequential(
            nn.Conv2d(128, 96, 1), nn.GroupNorm(12, 96), nn.GELU(),
            nn.Conv2d(96, 64, 1),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> DecoderOutput:
        semantic = F.interpolate(self.sam_head(features), image.shape[-2:], mode="bilinear", align_corners=False)
        app, shadow_logits = self.appearance(image)
        shadow_prob = shadow_logits.sigmoid().detach() if not self.training else shadow_logits.sigmoid()
        # Keep edge information from the appearance branch, but reduce its
        # influence where the branch predicts strong illumination change.
        gated_app = app * (1.0 - 0.65 * shadow_prob)
        fused = self.fuse(torch.cat((semantic, gated_app), 1))
        logits = self.class_head(fused)
        presence_logits = self.presence_head(features)
        embedding = F.normalize(self.embedding_head(fused), dim=1)
        return DecoderOutput(logits, presence_logits, shadow_logits, embedding)


class FixedPointOnlySAM(nn.Module):
    sam_side = 1024

    def __init__(self, sam_source: str, checkpoint: str, classes: int,
                 use_lora: bool = True, lora_rank: int = 4,
                 lora_start_layer: int = 8):
        super().__init__()
        source = str(Path(sam_source).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        for p in self.sam.parameters():
            p.requires_grad = False
        if use_lora:
            install_lora(self.sam.image_encoder, lora_rank, 2.0 * lora_rank, lora_start_layer)
        self.decoder = FixedSemanticDecoder(classes)
        self.classes = classes

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(image, (self.sam_side, self.sam_side), mode="bilinear", align_corners=False)
        x = x * 255.0
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std
        if not any(p.requires_grad for p in self.sam.image_encoder.parameters()):
            with torch.no_grad():
                return self.sam.image_encoder(x)
        return self.sam.image_encoder(x)

    def semantic(self, features: torch.Tensor, image: torch.Tensor) -> DecoderOutput:
        return self.decoder(features, image)

    @torch.no_grad()
    def prompted_geometry(self, feature: torch.Tensor, points: torch.Tensor,
                          image_size: int, max_negative: int = 8) -> torch.Tensor:
        side = self.sam_side
        out = torch.full((self.classes, image_size, image_size), -10.0, device=feature.device)
        if len(points) == 0:
            return out
        p = points.to(feature.device)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        for class_id in p[:, 2].long().unique().tolist():
            positive = p[p[:, 2].long() == class_id, :2]
            other = p[p[:, 2].long() != class_id, :2]
            if len(other):
                distances = torch.cdist(other, positive).min(dim=1).values
                negative = other[distances.argsort()[:max_negative]]
            else:
                negative = other
            scale = side / float(image_size)
            coords = torch.cat((positive, negative), 0).unsqueeze(0) * scale
            labels = torch.cat((torch.ones(len(positive), device=feature.device),
                                torch.zeros(len(negative), device=feature.device))).long().unsqueeze(0)
            sparse, dense = self.sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=feature.unsqueeze(0), image_pe=dense_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            out[class_id] = low_res[0, 0]
        return F.interpolate(out.unsqueeze(0), (image_size, image_size), mode="bilinear", align_corners=False)[0]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.sam.state_dict().items()
                if ".a" in name or ".b" in name}

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            self.sam.load_state_dict(state, strict=False)
