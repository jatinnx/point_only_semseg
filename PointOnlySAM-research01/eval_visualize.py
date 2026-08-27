"""Evaluate a checkpoint and generate per-image visualizations.

All images are saved flat in one folder with naming:
  {idx:04d}_{image_id}_real.png
  {idx:04d}_{image_id}_gt.png
  {idx:04d}_{image_id}_bymodal.png
  {idx:04d}_{image_id}_overlay.png
  {idx:04d}_{image_id}_points.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from pointonlysam.data import PointOnlyDataset, collate
from pointonlysam.model import PointSAMSemantic
from pointonlysam.runtime import load_config

# ─── DLRSD 17-class colour palette (0-indexed) ────────────────────────────
# Derived from the DLRSD train_masks colour maps, mapped to model classes 0–16.
# Classes: 0=airplane  1=bare_soil  2=buildings  3=cars  4=chaparral
#          5=court  6=dock  7=field  8=grass  9=mobile_home
#         10=pavement  11=sand  12=sea  13=ship  14=tanks
#         15=trees  16=water
PALETTE = np.array([
    [192,   0,   0],  #  0 airplane       red
    [128,  64,   0],  #  1 bare_soil      brown
    [  0,   0, 192],  #  2 buildings      blue
    [  0, 192, 192],  #  3 cars           cyan
    [  0, 128,   0],  #  4 chaparral      green
    [128, 128,   0],  #  5 court          olive
    [128,   0, 128],  #  6 dock           purple
    [128, 192,   0],  #  7 field          lime-green
    [  0, 128, 128],  #  8 grass          teal
    [  0,  64, 128],  #  9 mobile_home    dark-blue
    [192, 128,   0],  # 10 pavement       orange
    [192, 192,   0],  # 11 sand           yellow
    [  0,   0, 128],  # 12 sea            navy
    [ 64,   0, 128],  # 13 ship           indigo
    [128,   0,  64],  # 14 tanks          maroon
    [  0, 128,  64],  # 15 trees          forest-green
    [  0, 192,   0],  # 16 water          bright-green
], dtype=np.uint8)

CLASS_NAMES = [
    "airplane", "bare_soil", "buildings", "cars", "chaparral",
    "court", "dock", "field", "grass", "mobile_home",
    "pavement", "sand", "sea", "ship", "tanks",
    "trees", "water",
]


# ─── Visualization helpers ────────────────────────────────────────────────

def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """(H, W) class ids → (H, W, 3) RGB using PALETTE."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(PALETTE)):
        rgb[mask == c] = PALETTE[c]
    return rgb


def make_overlay(real_rgb: np.ndarray, pred_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend prediction colour map over the real image."""
    return cv2.addWeighted(real_rgb, 1.0 - alpha, pred_rgb, alpha, 0)


def sample_points_from_gt(gt_mask: np.ndarray, num_per_class: int = 5,
                          rng: np.random.RandomState | None = None) -> np.ndarray:
    """Sample point annotations from a GT mask (mimics training data format).

    Returns (N, 3) array of [x, y, class] in pixel coordinates.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    points = []
    for c in np.unique(gt_mask):
        if c == 0:
            # background / void — skip or sample less
            pass
        ys, xs = np.where(gt_mask == c)
        if len(xs) == 0:
            continue
        n = min(num_per_class, len(xs))
        idx = rng.choice(len(xs), n, replace=False)
        for i in idx:
            points.append([int(xs[i]), int(ys[i]), int(c)])
    return np.array(points, dtype=np.int32) if points else np.zeros((0, 3), dtype=np.int32)


def draw_points(img_rgb: np.ndarray, points: np.ndarray, radius: int = 4) -> np.ndarray:
    """Draw coloured dots on a copy of the image."""
    vis = img_rgb.copy()
    for x, y, c in points:
        color = tuple(int(v) for v in PALETTE[c])
        cv2.circle(vis, (int(x), int(y)), radius, color, -1)
        cv2.circle(vis, (int(x), int(y)), radius, (255, 255, 255), 1)  # white border
    return vis


def make_legend() -> np.ndarray:
    """Create a legend image with class colours and names."""
    cell_h, cell_w, pad = 24, 180, 8
    total_h = len(CLASS_NAMES) * (cell_h + pad) + pad
    total_w = cell_w + pad * 2
    legend = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    for i, name in enumerate(CLASS_NAMES):
        y = pad + i * (cell_h + pad)
        color = tuple(int(v) for v in PALETTE[i])
        cv2.rectangle(legend, (pad, y), (pad + cell_h, y + cell_h), color, -1)
        cv2.rectangle(legend, (pad, y), (pad + cell_h, y + cell_h), (0, 0, 0), 1)
        cv2.putText(legend, f"{i:2d} {name}", (pad + cell_h + 8, y + cell_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return legend


# ─── Main evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def main(cfg: dict, checkpoint: str, out_dir: str, max_steps: int | None = None) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = PointOnlyDataset(cfg["val_manifest"], cfg["image_size"], training=False)
    loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                        collate_fn=collate)

    # Load model
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
                             use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
                             lora_start_layer=cfg.get("lora_start_layer", 0),
                             decoder_variant=cfg.get("decoder_variant", "rgb_fusion")).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(state.get("teacher_decoder", state["decoder"]))
    model.load_adapter_state(state.get("sam_lora", {}))
    print(f"Loaded checkpoint: {checkpoint}")
    print(f"Evaluating {len(dataset)} images → {out}")

    # Confusion matrix for mIoU
    C = cfg["num_classes"]
    matrix = torch.zeros(C, C, dtype=torch.int64, device=device)
    rng = np.random.RandomState(42)

    # Save legend once
    legend = make_legend()
    cv2.imwrite(str(out / "legend.png"), cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))

    for step, batch in enumerate(tqdm(loader, desc="eval"), start=1):
        image = batch["weak"].to(device)
        target = batch["mask"].to(device)
        img_id = batch["id"][0]

        # Forward pass
        logits = model.semantic(model.encode(image), image)
        pred = logits.argmax(1)  # (1, 256, 256)

        # Accumulate confusion matrix
        valid = (target >= 0) & (target < C)
        indices = C * target[valid] + pred[valid]
        matrix += torch.bincount(indices, minlength=C ** 2).reshape_as(matrix)

        # Convert to numpy for visualization
        pred_np = pred[0].cpu().numpy().astype(np.uint8)
        gt_np = target[0].cpu().numpy().astype(np.uint8)

        # Original real image (load from disk for full quality)
        item = dataset.items[step - 1]
        real_bgr = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        if real_bgr is not None:
            real_rgb = cv2.cvtColor(cv2.resize(real_bgr, (cfg["image_size"], cfg["image_size"]),
                                                interpolation=cv2.INTER_LINEAR),
                                    cv2.COLOR_BGR2RGB)
        else:
            real_rgb = (batch["weak"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Generate visualizations
        gt_rgb = mask_to_rgb(gt_np)
        pred_rgb = mask_to_rgb(pred_np)
        overlay = make_overlay(real_rgb, pred_rgb, alpha=0.45)
        points = sample_points_from_gt(gt_np, num_per_class=5, rng=rng)
        pts_img = draw_points(real_rgb, points, radius=5)

        # Save flat: {idx}_{id}_{type}.png
        prefix = f"{step:04d}_{img_id}"
        cv2.imwrite(str(out / f"{prefix}_real.png"), cv2.cvtColor(real_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{prefix}_gt.png"), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{prefix}_bymodal.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{prefix}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{prefix}_points.png"), cv2.cvtColor(pts_img, cv2.COLOR_RGB2BGR))

    # Compute metrics
    tp = matrix.diag().float()
    denom = matrix.sum(1).float() + matrix.sum(0).float() - tp
    iou = tp / denom.clamp_min(1)
    present = denom > 0
    accuracy = tp.sum() / matrix.sum().clamp_min(1)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {checkpoint}")
    print(f"{'='*60}")
    print(f"pixel_accuracy = {accuracy.item():.4f}")
    print(f"mIoU_present   = {iou[present].mean().item():.4f}")
    print(f"{'─'*60}")
    for class_id in range(C):
        iou_val = iou[class_id].item()
        marker = " ◄" if present[class_id] else " (absent)"
        print(f"  {class_id:2d} {CLASS_NAMES[class_id]:15s}  IoU = {iou_val:.4f}{marker}")
    print(f"{'='*60}")

    # Save metrics to file
    metrics_path = out / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"checkpoint: {checkpoint}\n")
        f.write(f"pixel_accuracy: {accuracy.item():.4f}\n")
        f.write(f"mIoU_present: {iou[present].mean().item():.4f}\n")
        for class_id in range(C):
            f.write(f"class_{class_id:02d}_{CLASS_NAMES[class_id]}_IoU: {iou[class_id].item():.4f}\n")
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Visualizations saved to {out}/ ({len(dataset)} images × 5 types = {len(dataset)*5} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate + visualise point-only SAM")
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam.json")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output-dir", default="runs/dlrsd_pointonly_sam_v1/eval_epoch030",
                        help="Directory for per-image visualizations")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Limit number of images (for smoke tests)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.checkpoint, args.output_dir, args.max_steps)
