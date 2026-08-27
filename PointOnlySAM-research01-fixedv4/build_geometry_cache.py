"""Build point-seeded, conflict-filtered SAM region targets without dense masks.

This is deliberately a separate, auditable stage. It reads only the point
training manifest, image pixels, and the public SAM checkpoint. It never opens
or references a dense training label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pointonlysam.data import PointOnlyDataset, collate
from pointonlysam.model import PointSAMSemantic
from pointonlysam.runtime import load_config


@torch.no_grad()
def main(cfg: dict, output: str, max_images: int | None = None) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    # training=False means no augmentation; manifest has no mask key and no
    # dense label is read. Points are still returned when present.
    data = PointOnlyDataset(cfg["train_manifest"], cfg["image_size"], training=False)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
                             use_lora=False).to(device).eval()
    regions: dict[str, dict[str, torch.Tensor]] = {}
    accepted = 0
    for index, batch in enumerate(loader, start=1):
        image, points = batch["weak"].to(device), batch["points"][0].to(device)
        masks = model.prompted_geometry(model.encode(image)[0], points, cfg["max_negative_points"])
        probs = masks.sigmoid()
        top_prob, label = probs.max(0)
        if probs.shape[0] > 1:
            second = probs.topk(2, dim=0).values[1]
        else:
            second = torch.zeros_like(top_prob)
        # A pixel is accepted only if SAM makes a high-confidence claim and
        # no competing point-prompted class also claims it.  Remove the outer
        # one-pixel ring: boundaries remain unsupervised instead of noisy.
        candidate = (top_prob >= cfg["region_probability"]) & ((top_prob - second) >= cfg["region_margin"])
        interior = F.avg_pool2d(candidate.float()[None, None], 3, 1, 1)[0, 0] > 0.999
        target = torch.full_like(label, 255, dtype=torch.uint8)
        target[interior] = label[interior].to(torch.uint8)
        confidence = torch.zeros_like(top_prob, dtype=torch.float16)
        confidence[interior] = top_prob[interior].to(torch.float16)
        # Preserve every human point even if SAM fails at that pixel.
        for x, y, c in points.long().tolist():
            target[y.clamp(0, cfg["image_size"] - 1) if isinstance(y, torch.Tensor) else max(0, min(cfg["image_size"] - 1, y)),
                   x.clamp(0, cfg["image_size"] - 1) if isinstance(x, torch.Tensor) else max(0, min(cfg["image_size"] - 1, x))] = c
        # The list values above are normal ints. Set confidence separately to
        # make the point constraints maximally trusted.
        for x, y, _ in points.long().tolist():
            confidence[max(0, min(cfg["image_size"] - 1, y)), max(0, min(cfg["image_size"] - 1, x))] = 1.0
        regions[batch["id"][0]] = {"label": target.cpu(), "confidence": confidence.cpu()}
        accepted += int(interior.sum())
        if index % 25 == 0 or index == len(loader):
            print(f"geometry {index}/{len(loader)}; trusted pixels/image={accepted / index:.0f}")
        if max_images is not None and index >= max_images:
            break
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"num_classes": cfg["num_classes"], "image_size": cfg["image_size"], "regions": regions}, path)
    print(f"Saved {len(regions)} point-only SAM region targets to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_v2.json")
    parser.add_argument("--output", default="artifacts/dlrsd_sam_regions_v2.pt")
    parser.add_argument("--max-images", type=int, help="Smoke-test cap; never use this partial cache for training.")
    args = parser.parse_args()
    main(load_config(args.config), args.output, args.max_images)
