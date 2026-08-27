"""Evaluation for corrected PointOnlySAM with optional prediction visualization saving.

Modes:
  image_only: main research setting; no test-time points are used to constrain classes.
  point_set: optional ablation; validation point labels are used only to suppress
             classes not represented by points.

With --save-preds DIR, the evaluator writes, per image:
  *_real.png
  *_gt.png
  *_bymodal.png
  *_overlay.png
  *_points.png
plus legend.png and metrics.txt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from pointonlysam.fixed_data import FixedPointOnlyDataset, collate_fixed
from pointonlysam.fixed_model import FixedPointOnlySAM
from pointonlysam.fixed_objectives import gate_logits, MultiPrototypeBank
from pointonlysam.runtime import load_config


CLASS_NAMES = [
    "airplane", "bare_soil", "buildings", "cars", "chaparral", "court",
    "dock", "field", "grass", "mobile_home", "pavement", "sand",
    "sea", "ship", "tanks", "trees", "water",
]

# Exact class colors from the supplied epoch-040 legend.png.
PALETTE = np.array([
    [192, 0, 0],
    [128, 64, 0],
    [0, 0, 192],
    [0, 192, 192],
    [0, 128, 0],
    [128, 128, 0],
    [128, 0, 128],
    [128, 192, 0],
    [0, 128, 128],
    [0, 64, 128],
    [192, 128, 0],
    [192, 192, 0],
    [0, 0, 128],
    [64, 0, 128],
    [128, 0, 64],
    [0, 128, 64],
    [0, 192, 0],
], dtype=np.uint8)


def colorize(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")
    return PALETTE[np.clip(mask.astype(np.int64), 0, len(PALETTE) - 1)]


def overlay(rgb: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    mask_f = colored_mask.astype(np.float32)
    out = (1.0 - alpha) * rgb_f + alpha * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_points(rgb: np.ndarray, points: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    for x, y, cls in points.astype(np.float32):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < out.shape[1] and 0 <= yi < out.shape[0]):
            continue
        c = tuple(int(v) for v in PALETTE[int(cls) % len(PALETTE)].tolist())
        # black outer ring + class-colored inner dot
        cv2.circle(out, (xi, yi), 5, (0, 0, 0), -1)
        cv2.circle(out, (xi, yi), 3, c, -1)
    return out


def write_legend(path: Path) -> None:
    width = 320
    row_h = 32
    top = 10
    height = top + row_h * len(CLASS_NAMES) + 10
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for i, name in enumerate(CLASS_NAMES):
        y0 = top + i * row_h + 5
        canvas[y0:y0 + 22, 8:32] = PALETTE[i]
        cv2.putText(
            canvas,
            f"{i} {name}",
            (42, y0 + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def apply_point_classset(
    logits: torch.Tensor,
    points: list[torch.Tensor],
    scale: float = 8.0,
) -> torch.Tensor:
    gated = logits.clone()
    for i, p in enumerate(points):
        active = torch.zeros(logits.shape[1], dtype=torch.bool, device=logits.device)
        if len(p):
            active[p[:, 2].long().clamp(0, logits.shape[1] - 1).unique()] = True
        gated[i, ~active] -= scale
    return gated


@torch.no_grad()
def main(
    cfg: dict,
    checkpoint: str,
    mode: str,
    max_steps: int | None = None,
    save_preds: str | None = None,
) -> None:
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    dataset = FixedPointOnlyDataset(
        cfg["val_manifest"],
        cfg["image_size"],
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["eval_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate_fixed,
    )

    model = FixedPointOnlySAM(
        cfg["sam_source"],
        cfg["sam_checkpoint"],
        cfg["num_classes"],
        use_lora=cfg["train_sam_lora"],
        lora_rank=cfg["lora_rank"],
        lora_start_layer=cfg["lora_start_layer"],
    ).to(device).eval()

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.decoder.load_state_dict(state["teacher_decoder"])
    model.load_adapter_state(state.get("sam_lora", {}))

    bank = MultiPrototypeBank(
        cfg["num_classes"],
        256,
        device,
        cfg["prototypes_per_class"],
        cfg["prototype_momentum"],
        cfg["prototype_temperature"],
    )
    bank.load_state_dict(state["prototypes"])

    save_dir = Path(save_preds) if save_preds else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        write_legend(save_dir / "legend.png")

    matrix = torch.zeros(
        cfg["num_classes"],
        cfg["num_classes"],
        dtype=torch.int64,
        device=device,
    )

    for step, batch in enumerate(tqdm(loader, desc=f"evaluation-{mode}"), start=1):
        image = batch["weak"].to(device)
        target = batch["mask"].to(device)
        features = model.encode(image)
        out = model.semantic(features, image)

        logits, _ = gate_logits(
            out.logits,
            out.presence_logits,
            cfg["presence_threshold"],
            cfg["presence_gate_scale"],
            cfg["presence_top_k"],
        )

        proto = bank.scores(features)
        if proto is not None:
            proto_prob = F.interpolate(
                proto.probabilities,
                logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits = logits + cfg["inference_proto_alpha"] * torch.log(proto_prob.clamp_min(1e-5))

        if mode == "point_set":
            logits = apply_point_classset(
                logits,
                [p.to(device) for p in batch["points"]],
            )

        pred = logits.argmax(1)
        valid = (target >= 0) & (target < cfg["num_classes"])
        idx = cfg["num_classes"] * target[valid] + pred[valid]
        matrix += torch.bincount(
            idx,
            minlength=cfg["num_classes"] ** 2,
        ).reshape_as(matrix)

        if save_dir is not None:
            for bi, image_id in enumerate(batch["id"]):
                # weak is already the evaluation RGB image in [0,1].
                real = (
                    image[bi].detach().cpu().permute(1, 2, 0).numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
                gt = target[bi].detach().cpu().numpy().astype(np.int64)
                pr = pred[bi].detach().cpu().numpy().astype(np.int64)

                gt_rgb = colorize(gt)
                pr_rgb = colorize(pr)
                ov_rgb = overlay(real, pr_rgb, alpha=0.45)
                pts = batch["points"][bi].detach().cpu().numpy()
                pts_rgb = draw_points(real, pts)

                cv2.imwrite(
                    str(save_dir / f"{image_id}_real.png"),
                    cv2.cvtColor(real, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(save_dir / f"{image_id}_gt.png"),
                    cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(save_dir / f"{image_id}_bymodal.png"),
                    cv2.cvtColor(pr_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(save_dir / f"{image_id}_overlay.png"),
                    cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(save_dir / f"{image_id}_points.png"),
                    cv2.cvtColor(pts_rgb, cv2.COLOR_RGB2BGR),
                )

        if max_steps is not None and step >= max_steps:
            break

    tp = matrix.diag().float()
    denom = matrix.sum(1).float() + matrix.sum(0).float() - tp
    iou = tp / denom.clamp_min(1)
    present = denom > 0
    accuracy = tp.sum() / matrix.sum().clamp_min(1)

    metrics_lines = [
        f"checkpoint: {checkpoint}",
        f"mode: {mode}",
        f"pixel_accuracy: {accuracy.item():.4f}",
        f"mIoU_present: {iou[present].mean().item():.4f}",
    ]
    print(f"mode={mode}")
    print(f"pixel_accuracy={accuracy.item():.4f}")
    print(f"mIoU_present={iou[present].mean().item():.4f}")
    for c, value in enumerate(iou.tolist()):
        name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c)
        line = f"class_{c:02d}_{name}_IoU={value:.4f}"
        print(line)
        metrics_lines.append(line)

    if save_dir is not None:
        (save_dir / "metrics.txt").write_text("\n".join(metrics_lines) + "\n")
        print(f"Saved predictions and visualizations to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_fixed.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("image_only", "point_set"), default="image_only")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--save-preds",
        metavar="DIR",
        help="Save *_real.png, *_gt.png, *_bymodal.png, *_overlay.png, *_points.png, legend.png and metrics.txt.",
    )
    args = parser.parse_args()
    main(
        load_config(args.config),
        args.checkpoint,
        args.mode,
        args.max_steps,
        args.save_preds,
    )
