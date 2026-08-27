"""Generate CLEAN point-only manifest with ZERO dense mask dependency.

This script creates train.json using ONLY the point annotation masks.
No dense masks are read, referenced, or used in any way.

Usage:
    cd /home/cse-sdpl/Downloads/point_only_semseg
    python pointonlySAM-training-dataset/make_clean_manifest.py
"""
import json
import os
import numpy as np
import cv2
from pathlib import Path

# Paths
BASE_DIR = Path("/home/cse-sdpl/Downloads/point_only_semseg/pointonlySAM-training-dataset")
TRAIN_IMAGES = BASE_DIR / "train_images"
POINT_MASKS = BASE_DIR / "train_point_masks"
TEST_IMAGES = BASE_DIR / "test_images"
TEST_MASKS = BASE_DIR / "test_masks"

# DLRSD class mapping (1-17 in point masks -> 0-16 in model)
NUM_CLASSES = 17
POINTS_PER_CLASS = 5
GRID = 3
SEED = 42


def grid_pick(ys, xs, n, rng, grid=GRID):
    """Pick n well-spread points using grid-based spatial selection."""
    h, w = 256, 256
    if len(ys) <= n:
        return np.stack([ys, xs], 1)
    
    cy = np.clip(ys // (h // grid), 0, grid - 1)
    cx = np.clip(xs // (w // grid), 0, grid - 1)
    cells = {}
    for i in range(len(ys)):
        cells.setdefault((int(cy[i]), int(cx[i])), []).append(i)
    
    ordered = sorted(cells.items(), key=lambda kv: -len(kv[1]))
    chosen = []
    for (key, idxs) in ordered:
        if len(chosen) >= n:
            break
        sub_y, sub_x = ys[idxs], xs[idxs]
        cyc, cxc = sub_y.mean(), sub_x.mean()
        d = (sub_y - cyc) ** 2 + (sub_x - cxc) ** 2
        chosen.append((sub_y[int(d.argmin())], sub_x[int(d.argmin())]))
    
    if len(chosen) < n:
        used = set(chosen)
        rest = [(int(y), int(x)) for y, x in zip(ys, xs) if (int(y), int(x)) not in used]
        rng.shuffle(rest)
        chosen.extend(rest[:n - len(chosen)])
    
    return np.asarray(chosen, dtype=np.int64)


def extract_points_from_mask(point_mask_path, per_class=POINTS_PER_CLASS):
    """Extract points ONLY from point annotation mask. NO dense mask used."""
    pm = cv2.imread(str(point_mask_path), cv2.IMREAD_GRAYSCALE)
    if pm is None:
        raise FileNotFoundError(f"Cannot read: {point_mask_path}")
    
    points = []
    rng = np.random.default_rng(SEED + hash(str(point_mask_path)) % 100000)
    
    for c in range(1, NUM_CLASSES + 1):
        ys, xs = np.where(pm == c)
        n = min(per_class, len(xs))
        if n == 0:
            continue
        yx = grid_pick(ys, xs, n, rng)
        for y, x in yx:
            points.append([int(x), int(y), c - 1])  # class 0-16
    
    return points


def build_train_manifest():
    """Build train.json using ONLY point masks. ZERO dense mask dependency."""
    images = sorted(TRAIN_IMAGES.glob("*.png"))
    pmasks = sorted(POINT_MASKS.glob("*.png"))
    
    assert len(images) == len(pmasks), f"Image/mask count mismatch: {len(images)} vs {len(pmasks)}"
    
    rows = []
    total_points = 0
    
    for img_path, pm_path in zip(images, pmasks):
        assert img_path.stem == pm_path.stem, f"Name mismatch: {img_path} vs {pm_path}"
        
        # Read image to get dimensions
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        
        # Extract points ONLY from point mask
        points = extract_points_from_mask(pm_path)
        total_points += len(points)
        
        rows.append({
            "id": img_path.stem,
            "image": str(img_path),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
            "points": points,
            # NO "mask" key - point-only contract
        })
    
    out_path = BASE_DIR / "train.json"
    out_path.write_text(json.dumps(rows, indent=1))
    
    print(f"✅ CLEAN train.json created:")
    print(f"   Items: {len(rows)}")
    print(f"   Total points: {total_points}")
    print(f"   Points per class max: {POINTS_PER_CLASS}")
    print(f"   Dense mask dependency: ZERO")
    
    # Verify no mask keys
    items_with_mask = [r for r in rows if "mask" in r]
    print(f"   Items with mask key: {len(items_with_mask)} (must be 0)")


def build_val_manifest():
    """Build val.json for evaluation (masks needed for metrics only)."""
    images = sorted(TEST_IMAGES.glob("*.png"))
    masks = sorted(TEST_MASKS.glob("*.png"))
    
    assert len(images) == len(masks), f"Test image/mask count mismatch: {len(images)} vs {len(masks)}"
    
    rows = []
    for img_path, mask_path in zip(images, masks):
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        
        # Remap mask from 1-17 to 0-16
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        
        remapped = (mask.astype(np.int32) - 1).astype(np.uint8)
        
        # Save remapped mask
        remapped_dir = BASE_DIR / "val_masks_remapped"
        remapped_dir.mkdir(exist_ok=True)
        remapped_path = remapped_dir / mask_path.name
        cv2.imwrite(str(remapped_path), remapped)
        
        rows.append({
            "id": img_path.stem,
            "image": str(img_path),
            "mask": str(remapped_path),
            "width": int(img.shape[1]),
            "height": int(img.shape[0]),
        })
    
    out_path = BASE_DIR / "val.json"
    out_path.write_text(json.dumps(rows, indent=1))
    
    print(f"✅ CLEAN val.json created:")
    print(f"   Items: {len(rows)}")
    print(f"   Purpose: Evaluation metrics only (mIoU, PA, etc.)")


if __name__ == "__main__":
    print("=" * 60)
    print("CLEAN POINT-ONLY DATASET MANIFEST GENERATOR")
    print("ZERO dense mask dependency")
    print("=" * 60)
    print()
    
    build_train_manifest()
    print()
    build_val_manifest()
    
    print()
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # Verify train.json
    with open(BASE_DIR / "train.json") as f:
        train_data = json.load(f)
    
    has_mask = any("mask" in item for item in train_data)
    print(f"Train manifest has 'mask' key: {has_mask} (must be False)")
    print(f"Train images: {len(train_data)}")
    print(f"Total training points: {sum(len(item['points']) for item in train_data)}")
    
    # Verify no dense masks in training directory
    dense_mask_files = list((BASE_DIR / "train_images").glob("*mask*"))
    print(f"Dense mask files in train_images: {len(dense_mask_files)} (must be 0)")
    
    print()
    print("✅ CLEAN DATASET READY - ZERO DENSE MASK CONTAMINATION")
