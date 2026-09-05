"""
Run PointOnlySAM-fixed inference on external RSI test images.
Saves predictions in a proper folder structure:
  separate_test_results/
    <image_name>/
      original.png       - the input image
      segmentation.png   - predicted class map (colored)
      overlay.png        - prediction overlaid on original
    summary.txt          - metrics and image list
    class_mapping.txt    - DLRSD class index to name mapping
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure the project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pointonlysam.fixed_model import FixedPointOnlySAM
from pointonlysam.fixed_objectives import gate_logits, MultiPrototypeBank
from pointonlysam.runtime import load_config

# DLRSD 17-class palette (same as evaluate_fixed.py)
PALETTE = np.array([
    [192, 0, 0], [128, 64, 0], [0, 0, 192], [0, 192, 192], [0, 128, 0],
    [128, 128, 0], [128, 0, 128], [128, 192, 0], [0, 128, 128], [0, 64, 128],
    [192, 128, 0], [192, 192, 0], [0, 0, 128], [64, 0, 128], [128, 0, 64],
    [0, 128, 64], [0, 192, 0],
], dtype=np.uint8)

DLRSD_CLASSES = [
    "airplane", "bare_soil", "buildings", "cars", "chaparral",
    "court", "dock", "field", "grass", "mobile_home",
    "pavement", "sand", "sea", "ship", "tanks",
    "trees", "water",
]


def collect_images(input_path: Path) -> list[Path]:
    """Collect all image files from input_path (file or directory)."""
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in exts)


@torch.no_grad()
def run_inference(
    cfg: dict,
    checkpoint: str,
    input_dir: str,
    output_dir: str,
    max_images: int = 0,
) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    model = FixedPointOnlySAM(
        cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
        use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
        lora_start_layer=cfg["lora_start_layer"],
    ).to(device).eval()

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(state["teacher_decoder"])
    model.load_adapter_state(state.get("sam_lora", {}))

    bank = MultiPrototypeBank(
        cfg["num_classes"], 256, device,
        cfg["prototypes_per_class"], cfg["prototype_momentum"],
        cfg["prototype_temperature"],
    )
    bank.load_state_dict(state["prototypes"])

    # Collect images
    images = collect_images(Path(input_dir))
    if max_images > 0:
        images = images[:max_images]
    print(f"Found {len(images)} images in {input_dir}")

    # Setup output
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Write class mapping
    with open(outdir / "class_mapping.txt", "w") as f:
        for i, name in enumerate(DLRSD_CLASSES):
            f.write(f"{i}: {name}\n")

    # Run inference
    summary_lines = []
    total_time = 0

    for idx, img_path in enumerate(images):
        t0 = time.time()

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  [SKIP] Cannot read: {img_path.name}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_size = (rgb.shape[1], rgb.shape[0])  # (W, H)

        # Resize to model input size
        small = cv2.resize(rgb, (cfg["image_size"], cfg["image_size"]),
                          interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Forward pass
        features = model.encode(image)
        dec = model.semantic(features, image)
        logits, presence = gate_logits(
            dec.logits, dec.presence_logits,
            cfg["presence_threshold"], cfg["presence_gate_scale"],
            cfg["presence_top_k"],
        )

        # Add prototype scores
        proto = bank.scores(features)
        if proto is not None:
            pp = torch.nn.functional.interpolate(
                proto.probabilities, logits.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            logits = logits + cfg["inference_proto_alpha"] * torch.log(pp.clamp_min(1e-5))

        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # Resize prediction back to original size
        pred_full = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)

        # Get dominant class
        unique, counts = np.unique(pred_full, return_counts=True)
        dominant_class = DLRSD_CLASSES[unique[np.argmax(counts)]]
        dominant_pct = 100.0 * counts.max() / counts.sum()

        # Generate outputs
        seg_colored = PALETTE[pred_full]
        overlay = cv2.addWeighted(rgb, 0.55, seg_colored, 0.45, 0)

        # Create per-image output directory
        stem = img_path.stem
        img_outdir = outdir / stem
        img_outdir.mkdir(exist_ok=True)

        # Save outputs
        cv2.imwrite(str(img_outdir / "original.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_outdir / "segmentation.png"),
                    cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_outdir / "overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        dt = time.time() - t0
        total_time += dt

        # Log
        line = (
            f"{idx+1:3d}/{len(images)}  {img_path.name:<40s}  "
            f"dominant={dominant_class:<16s} ({dominant_pct:.1f}%)  "
            f"time={dt:.2f}s"
        )
        print(line)
        summary_lines.append(line)

        # Free GPU memory
        del features, dec, logits, proto, image
        torch.cuda.empty_cache()

    # Write summary
    avg_time = total_time / max(len(images), 1)
    with open(outdir / "summary.txt", "w") as f:
        f.write(f"PointOnlySAM-Fixed External Evaluation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Images: {len(images)}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Avg per image: {avg_time:.2f}s\n")
        f.write(f"\n{'='*50}\n")
        f.write(f"Predictions (per image):\n")
        f.write(f"{'-'*50}\n")
        for line in summary_lines:
            f.write(f"{line}\n")

    print(f"\n{'='*50}")
    print(f"Done! {len(images)} images processed in {total_time:.1f}s")
    print(f"Average: {avg_time:.2f}s/image")
    print(f"Results saved to: {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointOnlySAM external RSI evaluation")
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_fixed.json")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--input", required=True, help="Input image dir or file")
    parser.add_argument("--output-dir", default="separate_test_results",
                        help="Output directory")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0=all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_inference(cfg, args.checkpoint, args.input, args.output_dir, args.max_images)
