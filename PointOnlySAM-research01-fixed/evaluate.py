"""Dense-mask evaluation.  This is intentionally separate from train.py."""
from __future__ import annotations

import argparse
from pathlib import Path

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


@torch.no_grad()
def main(cfg: dict, checkpoint: str, max_steps: int | None = None) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    # Evaluation-only dataset: its dense masks never enter train.py.
    dataset = PointOnlyDataset(cfg["val_manifest"], cfg["image_size"], training=False)
    loader = DataLoader(dataset, batch_size=cfg["eval_batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", collate_fn=collate)
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
                             use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
                             lora_start_layer=cfg.get("lora_start_layer", 0),
                             decoder_variant=cfg.get("decoder_variant", "rgb_fusion")).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    # EMA decoder is the teacher used to create pseudo labels during training.
    model.decoder.load_state_dict(state.get("teacher_decoder", state["decoder"]))
    model.load_adapter_state(state.get("sam_lora", {}))
    matrix = torch.zeros(cfg["num_classes"], cfg["num_classes"], dtype=torch.int64, device=device)
    for step, batch in enumerate(tqdm(loader, desc="evaluation"), start=1):
        image, target = batch["weak"].to(device), batch["mask"].to(device)
        pred = model.semantic(model.encode(image), image).argmax(1)
        valid = (target >= 0) & (target < cfg["num_classes"])
        indices = cfg["num_classes"] * target[valid] + pred[valid]
        matrix += torch.bincount(indices, minlength=cfg["num_classes"] ** 2).reshape_as(matrix)
        if max_steps is not None and step >= max_steps:
            break
    tp = matrix.diag().float()
    denom = matrix.sum(1).float() + matrix.sum(0).float() - tp
    iou = tp / denom.clamp_min(1)
    present = denom > 0
    accuracy = tp.sum() / matrix.sum().clamp_min(1)
    print(f"pixel_accuracy={accuracy.item():.4f}")
    print(f"mIoU_present={iou[present].mean().item():.4f}")
    for class_id, value in enumerate(iou.tolist()):
        print(f"class_{class_id:02d}_IoU={value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-steps", type=int, help="Smoke-test cap; omit for full held-out evaluation.")
    args = parser.parse_args()
    main(load_config(args.config), args.checkpoint, args.max_steps)
