"""Data loading and augmentations.

The training dataset deliberately accepts point annotations only.  A dense-mask
key in a training manifest is a hard error, preventing accidental supervision
leakage from the Chakraborty data release.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_rgb(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)


def _shadow(image: np.ndarray, strength: tuple[float, float], grid: int = 16) -> np.ndarray:
    """Apply a smooth, spatially varying synthetic cast shadow.

    It changes illumination while retaining object geometry and every point
    label, making teacher/student consistency a direct shadow-invariance cue.
    """
    h, w = image.shape[:2]
    small = np.random.normal(size=(grid, grid)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), grid / 3)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    lo, hi = strength
    attenuation = 1.0 - (lo + (hi - lo) * field)[..., None]
    return np.clip(image.astype(np.float32) * attenuation, 0, 255).astype(np.uint8)


class PointPairAugment:
    def __init__(self, size: int, shadow_probability: float = 0.75, static_regions: bool = False):
        self.size = size
        self.shadow_probability = shadow_probability
        self.static_regions = static_regions

    def __call__(self, image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # All spatial transforms are shared by weak/strong views so their
        # pixel coordinates remain aligned for consistency and pseudo labels.
        flip_x = False if self.static_regions else random.random() < 0.5
        flip_y = False if self.static_regions else random.random() < 0.5
        turns = 0 if self.static_regions else random.randrange(4)
        out = image.copy()
        pts = points.copy()
        h, w = out.shape[:2]
        if flip_x:
            out = out[:, ::-1].copy()
            pts[:, 0] = w - 1 - pts[:, 0]
        if flip_y:
            out = out[::-1].copy()
            pts[:, 1] = h - 1 - pts[:, 1]
        for _ in range(turns):
            out = np.rot90(out).copy()
            old_x = pts[:, 0].copy()
            pts[:, 0] = pts[:, 1]
            pts[:, 1] = w - 1 - old_x
            h, w = w, h

        weak = out.astype(np.float32) / 255.0
        weak = np.clip(weak + random.uniform(-0.03, 0.03), 0, 1)
        strong = out
        if random.random() < self.shadow_probability:
            strong = _shadow(strong, (0.12, 0.48))
        strong = strong.astype(np.float32) / 255.0
        contrast = random.uniform(0.75, 1.25)
        strong = np.clip((strong - 0.5) * contrast + 0.5, 0, 1)
        strong = np.clip(strong + np.random.normal(0, 0.015, strong.shape), 0, 1)
        return weak, strong.astype(np.float32), pts


class PointOnlyDataset(Dataset):
    def __init__(self, manifest: str, image_size: int, training: bool, geometry_cache: str | None = None):
        self.items: list[dict[str, Any]] = json.loads(Path(manifest).read_text())
        self.image_size = image_size
        self.training = training
        if training:
            leaks = [str(x.get("id", i)) for i, x in enumerate(self.items) if "mask" in x]
            if leaks:
                raise ValueError(
                    "Point-only contract violated: training manifest contains dense masks "
                    f"(first id: {leaks[0]})."
                )
            if not all("points" in x for x in self.items):
                raise ValueError("Training manifest must provide point annotations for every image.")
        self.regions = None
        if geometry_cache:
            if not training:
                raise ValueError("Geometry cache is a training-only pseudo-label source.")
            cache = torch.load(geometry_cache, map_location="cpu", weights_only=False)
            if int(cache["num_classes"]) <= 1 or int(cache["image_size"]) != image_size:
                raise ValueError("Geometry cache is incompatible with this configuration.")
            self.regions = cache["regions"]
            absent = [str(x["id"]) for x in self.items if str(x["id"]) not in self.regions]
            if absent:
                raise ValueError(f"Geometry cache misses training image id {absent[0]}.")
        self.augment = PointPairAugment(image_size, static_regions=self.regions is not None) if training else None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = _read_rgb(item["image"], self.image_size)
        points = np.asarray(item.get("points", []), dtype=np.float32).reshape(-1, 3)
        if len(points):
            points[:, 0] *= self.image_size / float(item.get("width", self.image_size))
            points[:, 1] *= self.image_size / float(item.get("height", self.image_size))
        if self.augment:
            weak, strong, points = self.augment(image, points)
        else:
            weak = image.astype(np.float32) / 255.0
            strong = weak.copy()
        result: dict[str, Any] = {
            "weak": torch.from_numpy(weak).permute(2, 0, 1).float(),
            "strong": torch.from_numpy(strong).permute(2, 0, 1).float(),
            "points": torch.from_numpy(points).float(),
            "id": str(item.get("id", index)),
        }
        if self.regions is not None:
            record = self.regions[str(item["id"])]
            result["region_label"] = record["label"].long()
            result["region_confidence"] = record["confidence"].float()
        # This branch is reachable only when training=False, which is used by
        # evaluate.py.  train.py never constructs a validation dataset.
        if not self.training and "mask" in item:
            mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(item["mask"])
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            result["mask"] = torch.from_numpy(mask.astype(np.int64))
        return result


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "weak": torch.stack([x["weak"] for x in batch]),
        "strong": torch.stack([x["strong"] for x in batch]),
        "points": [x["points"] for x in batch],
        "id": [x["id"] for x in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([x["mask"] for x in batch])
    if "region_label" in batch[0]:
        out["region_label"] = torch.stack([x["region_label"] for x in batch])
        out["region_confidence"] = torch.stack([x["region_confidence"] for x in batch])
    return out
