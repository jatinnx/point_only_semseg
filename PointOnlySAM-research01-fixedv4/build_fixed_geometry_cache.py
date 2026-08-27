"""Build point-only SAM pseudo regions + 2-D boundary targets.

No dense masks are read. The cache is derived only from RGB images, point
annotations, and the public SAM checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointonlysam.fixed_data import FixedPointOnlyDataset, collate_fixed
from pointonlysam.fixed_model import FixedPointOnlySAM
from pointonlysam.runtime import load_config


def make_boundary_targets(label: np.ndarray, known: np.ndarray, num_classes: int):
    boundary = np.zeros_like(known, dtype=np.uint8)
    # Object/region boundaries for every prompted class. This is intentionally
    # derived from SAM point prompts, not dense ground truth.
    for c in range(num_classes):
        mask = ((label == c) & known).astype(np.uint8)
        if mask.sum() == 0:
            continue
        dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        ero = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary |= (dil > ero).astype(np.uint8)
    support = cv2.dilate(boundary, np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    # Regions well inside the confident SAM labels receive smoothness support;
    # this suppresses artificial micro-boundaries without blurring edges.
    interior = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    return torch.from_numpy(boundary.astype(np.float16)), torch.from_numpy(support), torch.from_numpy(interior)


@torch.no_grad()
def main(cfg: dict, output: str, max_images: int | None = None) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    data = FixedPointOnlyDataset(cfg["train_manifest"], cfg["image_size"], training=False)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_fixed)
    model = FixedPointOnlySAM(
        cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
        use_lora=False,
    ).to(device).eval()
    regions = {}
    accepted = 0
    for index, batch in enumerate(loader, start=1):
        image = batch["weak"].to(device)
        points = batch["points"][0].to(device)
        feature = model.encode(image)[0]
        masks = model.prompted_geometry(feature, points, cfg["image_size"], cfg["max_negative_points"])
        probs = masks.sigmoid()
        top_prob, label = probs.max(0)
        if probs.shape[0] > 1:
            second = probs.topk(2, dim=0).values[1]
        else:
            second = torch.zeros_like(top_prob)
        known = (top_prob >= cfg["region_probability"]) & ((top_prob - second) >= cfg["region_margin"])
        interior = F.avg_pool2d(known.float()[None, None], 3, 1, 1)[0, 0] > 0.999
        target = torch.full_like(label, 255, dtype=torch.long)
        target[interior] = label[interior].long()
        confidence = torch.zeros_like(top_prob, dtype=torch.float32)
        confidence[interior] = top_prob[interior]
        for x, y, c in points.long().tolist():
            xx = max(0, min(cfg["image_size"] - 1, x))
            yy = max(0, min(cfg["image_size"] - 1, y))
            target[yy, xx] = c
            confidence[yy, xx] = 1.0
            known[yy, xx] = True
            label[yy, xx] = c
        boundary, support, smooth = make_boundary_targets(
            label.detach().cpu().numpy().astype(np.int16),
            known.detach().cpu().numpy().astype(bool),
            cfg["num_classes"],
        )
        regions[batch["id"][0]] = {
            "label": target.cpu().to(torch.uint8),
            "confidence": confidence.cpu().half(),
            "boundary_target": boundary,
            "boundary_support": support,
            "smooth_support": smooth,
        }
        accepted += int(interior.sum())
        if index % 25 == 0 or index == len(loader):
            print(f"geometry {index}/{len(loader)}; trusted pixels/image={accepted / index:.0f}")
        if max_images is not None and index >= max_images:
            break
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"num_classes": cfg["num_classes"], "image_size": cfg["image_size"], "regions": regions}, path)
    print(f"Saved {len(regions)} point-only geometry records to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_fixed.json")
    parser.add_argument("--output", default="artifacts/dlrsd_sam_regions_fixed.pt")
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    main(load_config(args.config), args.output, args.max_images)
