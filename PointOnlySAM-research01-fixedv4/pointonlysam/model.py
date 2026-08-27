"""Frozen SAM geometry backbone, optional LoRA, and semantic decoder."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)
        for p in base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.a.t() @ self.b.t()) * self.scale


def _install_lora(image_encoder: nn.Module, rank: int, alpha: float, start_layer: int = 0) -> None:
    for p in image_encoder.parameters():
        p.requires_grad = False
    for index, block in enumerate(image_encoder.blocks):
        if index < start_layer:
            continue
        block.attn.qkv = LoRALinear(block.attn.qkv, rank, alpha)
        block.attn.proj = LoRALinear(block.attn.proj, rank, alpha)


class SemanticDecoder(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(256, 192, 3, padding=1), nn.GroupNorm(24, 192), nn.GELU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
        )
        # SAM's 64x64 embedding is semantically useful but cannot by itself
        # represent 1-3 pixel roofs, vehicles, or crisp shadow boundaries.
        # This branch restores image-scale texture while SAM provides context.
        self.rgb = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(160, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, classes, 1),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        semantic = F.interpolate(self.head(features), image.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((semantic, self.rgb(image)), 1))


class BaselineSemanticDecoder(nn.Module):
    """The v1-style smooth decoder: SAM features only, no raw-RGB shortcut."""
    def __init__(self, classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.GroupNorm(32, 256), nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, classes, 1),
        )

    def forward(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.head(features), image.shape[-2:], mode="bilinear", align_corners=False)


class PointSAMSemantic(nn.Module):
    """Dense semantic decoder with SAM used only as a class-agnostic shape prior."""
    sam_side = 1024

    def __init__(self, sam_source: str, checkpoint: str, classes: int,
                 use_lora: bool = False, lora_rank: int = 4,
                 lora_start_layer: int = 0, decoder_variant: str = "rgb_fusion"):
        super().__init__()
        source = str(Path(sam_source).resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        from segment_anything import sam_model_registry
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)
        for p in self.sam.parameters():
            p.requires_grad = False
        if use_lora:
            _install_lora(self.sam.image_encoder, lora_rank, 2.0 * lora_rank, lora_start_layer)
        if decoder_variant == "baseline":
            self.decoder = BaselineSemanticDecoder(classes)
        elif decoder_variant == "rgb_fusion":
            self.decoder = SemanticDecoder(classes)
        else:
            raise ValueError(f"Unknown decoder_variant: {decoder_variant}")
        self.classes = classes

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """SAM-normalized 256px image -> cached 64x64 feature map."""
        x = F.interpolate(image, (self.sam_side, self.sam_side), mode="bilinear", align_corners=False)
        x = x * 255.0
        x = (x - self.sam.pixel_mean) / self.sam.pixel_std
        # Retaining a graph through frozen ViT costs substantial VRAM without
        # benefiting the decoder.  LoRA is an explicit opt-in experiment.
        if not any(p.requires_grad for p in self.sam.image_encoder.parameters()):
            with torch.no_grad():
                return self.sam.image_encoder(x)
        return self.sam.image_encoder(x)

    def semantic(self, features: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(features, image)

    @torch.no_grad()
    def prompted_geometry(self, feature: torch.Tensor, points: torch.Tensor,
                          max_negative: int = 8) -> torch.Tensor:
        """Class-conditioned SAM masks from human point labels.

        A class gets its own positive clicks.  Nearest points of all other
        classes are negative clicks—the NPC idea transferred to semantic data.
        Unprompted classes remain unclaimed (-inf-like logits), so SAM never
        invents a semantic class.
        """
        side, device = self.sam_side, feature.device
        out = torch.full((self.classes, 256, 256), -10.0, device=device)
        if len(points) == 0:
            return out
        p = points.to(device)
        dense_pe = self.sam.prompt_encoder.get_dense_pe()
        for class_id in p[:, 2].long().unique().tolist():
            positive = p[p[:, 2].long() == class_id, :2]
            other = p[p[:, 2].long() != class_id, :2]
            if len(other):
                distances = torch.cdist(other, positive).min(dim=1).values
                negative = other[distances.argsort()[:max_negative]]
            else:
                negative = other
            coords = torch.cat((positive, negative), 0).unsqueeze(0) * (side / 256.0)
            labels = torch.cat((torch.ones(len(positive), device=device),
                                torch.zeros(len(negative), device=device))).long().unsqueeze(0)
            sparse, dense = self.sam.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
            low_res, _ = self.sam.mask_decoder(
                image_embeddings=feature.unsqueeze(0), image_pe=dense_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            out[class_id] = low_res[0, 0]
        return F.interpolate(out.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False)[0]

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().cpu() for name, tensor in self.sam.state_dict().items()
                if ".a" in name or ".b" in name}

    def load_adapter_state(self, state: dict[str, torch.Tensor]) -> None:
        if state:
            self.sam.load_state_dict(state, strict=False)
