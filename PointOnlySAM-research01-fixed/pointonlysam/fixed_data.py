"""Point-only data and synthetic shadow metadata for the corrected model."""
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


def _shadow(image: np.ndarray, strength: tuple[float, float], grid: int = 16) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    small = np.random.normal(size=(grid, grid)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), grid / 3)
    field = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    field = (field - field.min()) / (field.max() - field.min() + 1e-6)
    lo, hi = strength
    attenuation = lo + (hi - lo) * field
    out = np.clip(image.astype(np.float32) * (1.0 - attenuation[..., None]), 0, 255).astype(np.uint8)
    # Soft synthetic-shadow confidence map. Zero means no synthetic shadow.
    return out, attenuation.astype(np.float32)


class FixedPointPairAugment:
    def __init__(self, size: int, shadow_probability: float = 0.75):
        self.size = size
        self.shadow_probability = shadow_probability

    def __call__(self, image: np.ndarray, points: np.ndarray):
        flip_x = random.random() < 0.5
        flip_y = random.random() < 0.5
        turns = random.randrange(4)
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

        weak = np.clip(out.astype(np.float32) / 255.0 + random.uniform(-0.02, 0.02), 0, 1)
        strong = out
        shadow_mask = np.zeros((self.size, self.size), dtype=np.float32)
        if random.random() < self.shadow_probability:
            strong, attenuation = _shadow(strong, (0.10, 0.52))
            # Only moderately/strongly attenuated regions receive the auxiliary
            # semantic invariance target; the attenuation map is not a semantic label.
            shadow_mask = np.clip((attenuation - 0.08) / 0.32, 0, 1)
        strong = strong.astype(np.float32) / 255.0
        contrast = random.uniform(0.80, 1.20)
        strong = np.clip((strong - 0.5) * contrast + 0.5, 0, 1)
        strong = np.clip(strong + np.random.normal(0, 0.012, strong.shape), 0, 1)
        return weak, strong.astype(np.float32), pts, shadow_mask


class FixedPointOnlyDataset(Dataset):
    def __init__(self, manifest: str, image_size: int, training: bool,
                 geometry_cache: str | None = None, shadow_probability: float = 0.75):
        self.items: list[dict[str, Any]] = json.loads(Path(manifest).read_text())
        self.image_size = image_size
        self.training = training
        if training:
            leaks = [str(x.get("id", i)) for i, x in enumerate(self.items) if "mask" in x]
            if leaks:
                raise ValueError(f"Point-only contract violated by dense mask in {leaks[0]}")
            if not all("points" in x for x in self.items):
                raise ValueError("Training manifest must provide point annotations for every image.")
        self.regions = None
        if geometry_cache:
            if not training:
                raise ValueError("Geometry cache is training-only.")
            cache = torch.load(geometry_cache, map_location="cpu", weights_only=False)
            if int(cache["num_classes"]) <= 1 or int(cache["image_size"]) != image_size:
                raise ValueError("Incompatible geometry cache")
            self.regions = cache["regions"]
            absent = [str(x["id"]) for x in self.items if str(x["id"]) not in self.regions]
            if absent:
                raise ValueError(f"Geometry cache misses {absent[0]}")
        self.augment = FixedPointPairAugment(image_size, shadow_probability=shadow_probability) if training else None

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
            weak, strong, points, shadow_mask = self.augment(image, points)
        else:
            weak = image.astype(np.float32) / 255.0
            strong = weak.copy()
            shadow_mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        result: dict[str, Any] = {
            "weak": torch.from_numpy(weak).permute(2, 0, 1).float(),
            "strong": torch.from_numpy(strong).permute(2, 0, 1).float(),
            "shadow_mask": torch.from_numpy(shadow_mask).float(),
            "points": torch.from_numpy(points).float(),
            "id": str(item.get("id", index)),
        }
        if self.regions is not None:
            record = self.regions[str(item["id"])]
            result["region_label"] = record["label"].long()
            result["region_confidence"] = record["confidence"].float()
            result["boundary_target"] = record["boundary_target"].float()
            result["boundary_support"] = record["boundary_support"].bool()
            result["smooth_support"] = record["smooth_support"].bool()
        if not self.training and "mask" in item:
            mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(item["mask"])
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            result["mask"] = torch.from_numpy(mask.astype(np.int64))
        return result


def collate_fixed(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "weak": torch.stack([x["weak"] for x in batch]),
        "strong": torch.stack([x["strong"] for x in batch]),
        "shadow_mask": torch.stack([x["shadow_mask"] for x in batch]),
        "points": [x["points"] for x in batch],
        "id": [x["id"] for x in batch],
    }
    for key in ("mask", "region_label", "region_confidence", "boundary_target", "boundary_support", "smooth_support"):
        if key in batch[0]:
            out[key] = torch.stack([x[key] for x in batch])
    return out
