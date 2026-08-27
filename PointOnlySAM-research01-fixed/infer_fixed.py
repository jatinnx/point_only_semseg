"""Image-only inference for the corrected PointOnlySAM model."""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from pointonlysam.fixed_model import FixedPointOnlySAM
from pointonlysam.fixed_objectives import gate_logits, MultiPrototypeBank
from pointonlysam.runtime import load_config

PALETTE = np.array([
    [192, 0, 0], [128, 64, 0], [0, 0, 192], [0, 192, 192], [0, 128, 0],
    [128, 128, 0], [128, 0, 128], [128, 192, 0], [0, 128, 128], [0, 64, 128],
    [192, 128, 0], [192, 192, 0], [0, 0, 128], [64, 0, 128], [128, 0, 64],
    [0, 128, 64], [0, 192, 0],
], dtype=np.uint8)


def files(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})


@torch.no_grad()
def main(cfg: dict, checkpoint: str, input_path: str, output_dir: str) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    model = FixedPointOnlySAM(
        cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
        use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"], lora_start_layer=cfg["lora_start_layer"],
    ).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(state["teacher_decoder"])
    model.load_adapter_state(state.get("sam_lora", {}))
    bank = MultiPrototypeBank(
        cfg["num_classes"], 256, device,
        cfg["prototypes_per_class"], cfg["prototype_momentum"], cfg["prototype_temperature"],
    )
    bank.load_state_dict(state["prototypes"])
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = files(Path(input_path))
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        original = (rgb.shape[1], rgb.shape[0])
        small = cv2.resize(rgb, (cfg["image_size"], cfg["image_size"]), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(small).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        features = model.encode(image)
        dec = model.semantic(features, image)
        logits, presence = gate_logits(
            dec.logits, dec.presence_logits,
            cfg["presence_threshold"], cfg["presence_gate_scale"], cfg["presence_top_k"],
        )
        proto = bank.scores(features)
        if proto is not None:
            pp = torch.nn.functional.interpolate(proto.probabilities, logits.shape[-2:], mode="bilinear", align_corners=False)
            logits = logits + cfg["inference_proto_alpha"] * torch.log(pp.clamp_min(1e-5))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred = cv2.resize(pred, original, interpolation=cv2.INTER_NEAREST)
        seg = PALETTE[pred]
        overlay = cv2.addWeighted(rgb, 0.55, seg, 0.45, 0)
        cv2.imwrite(str(outdir / f"{path.stem}_segmentation.png"), cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(outdir / f"{path.stem}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        np.save(outdir / f"{path.stem}_presence.npy", presence[0].cpu().numpy())
    print(f"Wrote corrected image-only predictions to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_fixed.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(load_config(args.config), args.checkpoint, args.input, args.output_dir)
