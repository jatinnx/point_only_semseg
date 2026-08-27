"""Image-only dense semantic inference for a trained PointOnlySAM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from pointonlysam.model import PointSAMSemantic
from pointonlysam.runtime import load_config

PALETTE = np.array([
    [192, 0, 0], [128, 64, 0], [0, 0, 192], [0, 192, 192], [0, 128, 0],
    [128, 128, 0], [128, 0, 128], [128, 192, 0], [0, 128, 128], [0, 64, 128],
    [192, 128, 0], [192, 192, 0], [0, 0, 128], [64, 0, 128], [128, 0, 64],
    [0, 128, 64], [0, 192, 0],
], dtype=np.uint8)


def inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@torch.no_grad()
def main(cfg: dict, checkpoint: str, input_path: str, output_dir: str) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
                             use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
                             lora_start_layer=cfg.get("lora_start_layer", 0),
                             decoder_variant=cfg.get("decoder_variant", "rgb_fusion")).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(state.get("teacher_decoder", state["decoder"]))
    model.load_adapter_state(state.get("sam_lora", {}))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = inputs(Path(input_path))
    if not files:
        raise FileNotFoundError(f"No supported images found under {input_path}")
    for path in files:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skipping unreadable file: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original_size = (rgb.shape[1], rgb.shape[0])
        resized = cv2.resize(rgb, (cfg["image_size"], cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        prediction = model.semantic(model.encode(image), image).argmax(1)[0].cpu().numpy().astype(np.uint8)
        # Preserve the input's dimensions in deliverable maps.
        prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)
        seg = PALETTE[prediction]
        overlay = cv2.addWeighted(rgb, 0.55, seg, 0.45, 0)
        stem = path.stem
        cv2.imwrite(str(output / f"{stem}_segmentation.png"), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / f"{stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Wrote image-only segmentation maps for {len(files)} input image(s) to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_v3a_lora.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="One RS image or a directory of images.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(load_config(args.config), args.checkpoint, args.input, args.output_dir)
