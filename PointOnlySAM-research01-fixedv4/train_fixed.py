"""Train the corrected PointOnlySAM semantic model using point labels only."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

from pointonlysam.fixed_data import FixedPointOnlyDataset, collate_fixed
from pointonlysam.fixed_model import FixedPointOnlySAM
from pointonlysam.fixed_objectives import (
    MultiPrototypeBank, boundary_loss_2d, conservative_pseudo, force_present_classes,
    gate_logits, partial_presence_bce, point_ce, region_ce,
    shadow_disentanglement_loss, shadow_mask_bce, weighted_pseudo_ce,
)
from pointonlysam.runtime import ema_update, load_config, make_teacher, seed_everything


def save_checkpoint(path: Path, model, teacher, prototypes, epoch: int, cfg: dict) -> None:
    torch.save({
        "epoch": epoch,
        "decoder": model.decoder.state_dict(),
        "teacher_decoder": teacher.state_dict(),
        "sam_lora": model.adapter_state(),
        "prototypes": prototypes.state_dict(),
        "config": cfg,
    }, path)


def main(cfg: dict, resume: str | None = None, max_steps: int | None = None) -> None:
    seed_everything(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    train_set = FixedPointOnlyDataset(
        cfg["train_manifest"], cfg["image_size"], training=True,
        geometry_cache=cfg.get("geometry_cache"),
        shadow_probability=cfg.get("shadow_probability", 0.75),
    )
    counts = torch.zeros(cfg["num_classes"])
    image_weights = []
    for item in train_set.items:
        ids = torch.tensor([int(p[2]) for p in item["points"]], dtype=torch.long)
        counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    class_weights = counts.clamp_min(1).rsqrt()
    class_weights = (class_weights / class_weights.mean()).clamp(max=4.0).to(device)
    for item in train_set.items:
        ids = torch.tensor([int(p[2]) for p in item["points"]], dtype=torch.long)
        image_weights.append(float(class_weights.cpu()[ids].mean()) if len(ids) else 1.0)
    sampler = WeightedRandomSampler(image_weights, num_samples=len(image_weights), replacement=True) \
        if cfg.get("class_balanced_sampling", True) else None
    loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=sampler is None, sampler=sampler,
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda", collate_fn=collate_fixed,
    )
    model = FixedPointOnlySAM(
        cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
        use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
        lora_start_layer=cfg["lora_start_layer"],
    ).to(device)
    teacher = make_teacher(model.decoder).to(device)
    prototypes = MultiPrototypeBank(
        cfg["num_classes"], 256, device,
        prototypes_per_class=cfg["prototypes_per_class"],
        momentum=cfg["prototype_momentum"],
        temperature=cfg["prototype_temperature"],
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.decoder.load_state_dict(state["decoder"])
        model.load_adapter_state(state.get("sam_lora", {}))
        teacher.load_state_dict(state["teacher_decoder"])
        prototypes.load_state_dict(state["prototypes"])
        start_epoch = int(state["epoch"]) + 1

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics_file = out / "metrics.jsonl"
    print(f"Fixed PointOnlySAM: {len(train_set)} images, point-only training, no dense train masks.")

    global_step = 0

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        teacher.eval()
        started = time.perf_counter()
        totals = {k: 0.0 for k in (
            "point", "region", "pseudo", "presence", "proto", "shadow", "shadow_head", "boundary", "smooth", "accepted"
        )}
        stop_training = False
        for step, batch in enumerate(tqdm(loader, desc=f"fixed epoch {epoch}/{cfg['epochs']}"), start=1):
            global_step += 1
            weak = batch["weak"].to(device)
            strong = batch["strong"].to(device)
            shadow_mask = batch["shadow_mask"].to(device)
            points = [x.to(device) for x in batch["points"]]
            with torch.no_grad():
                weak_features = model.encode(weak)
                # Update the bank with current human point features before using
                # it as a target. No gradients ever update prototype tensors.
                prototypes.update(weak_features, points, cfg["image_size"])
                teacher_out = teacher(weak_features, weak)
                teacher_presence = teacher_out.presence_logits.sigmoid()
                forced_teacher_presence = force_present_classes(
                    teacher_presence, points, cfg["num_classes"]
                )
                teacher_gated, teacher_presence = gate_logits(
                    teacher_out.logits, teacher_out.presence_logits,
                    cfg["presence_threshold"],
                    cfg["presence_gate_scale"],
                    cfg["presence_top_k"],
                    forced_presence=forced_teacher_presence,
                )
                proto = prototypes.scores(weak_features)
                proto_prob = None if proto is None else F.interpolate(
                    proto.probabilities, (cfg["image_size"], cfg["image_size"]), mode="bilinear", align_corners=False
                )
                region = batch["region_label"].to(device)
                region_known = region != 255
                # Reconstruct cached class-specific SAM geometry logits for the
                # pseudo-label gate. Unknown classes remain unavailable.
                sam = torch.full(
                    (len(points), cfg["num_classes"], cfg["image_size"], cfg["image_size"]),
                    -10.0, device=device,
                )
                class_index = region.clamp(0, cfg["num_classes"] - 1).unsqueeze(1)
                sam.scatter_(1, class_index, torch.where(
                    region_known, torch.full_like(region, 10.0), torch.full_like(region, -10.0)
                ).unsqueeze(1).float())
                sam_masks = [sam[i] for i in range(len(points))]
                labels, confidence, valid = conservative_pseudo(
                    teacher_gated, teacher_presence, sam_masks, proto_prob,
                    warm=epoch <= cfg["warmup_epochs"],
                    class_threshold=cfg["presence_threshold"],
                )
                teacher_conf = teacher_gated.softmax(1).max(1).values

                if max_steps is not None and global_step == 1:
                    print(
                        f"[DEBUG] teacher_conf_mean={teacher_conf.mean().item():.4f} "
                        f"teacher_conf_max={teacher_conf.max().item():.4f} "
                        f"region_known={region_known.float().mean().item():.4f} "
                        f"shadow_coverage={(shadow_mask > 0.20).float().mean().item():.4f}"
                    )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                strong_features = model.encode(strong)
                student_out = model.semantic(strong_features, strong)
                student_presence = student_out.presence_logits.sigmoid()
                forced_student_presence = force_present_classes(
                    student_presence, points, cfg["num_classes"]
                )
                student_gated, student_presence = gate_logits(
                    student_out.logits, student_out.presence_logits,
                    cfg["presence_threshold"],
                    cfg["presence_gate_scale"],
                    cfg["presence_top_k"],
                    forced_presence=forced_student_presence,
                )
                loss_point = point_ce(student_out.logits, points, class_weights)
                loss_region = region_ce(
                    student_gated, batch["region_label"].to(device),
                    batch["region_confidence"].to(device), class_weights,
                )
                loss_pseudo = weighted_pseudo_ce(student_gated, labels, confidence, valid)
                loss_presence = partial_presence_bce(
                    student_out.presence_logits,
                    points,
                    cfg["num_classes"],
                )
                loss_proto = prototypes.point_margin_loss(
                    strong_features, points, cfg["image_size"], cfg["prototype_margin"]
                )
                # shadow_valid = teacher_conf >= cfg.get("shadow_teacher_confidence", 0.50)
                loss_shadow = shadow_disentanglement_loss(
                    student_gated,
                    teacher_gated,
                    shadow_mask,
                    # shadow_valid,
                    None,
                    teacher_conf,
                )
                loss_shadow_head = shadow_mask_bce(student_out.shadow_logits, shadow_mask)
                loss_boundary, loss_smooth = boundary_loss_2d(
                    student_gated,
                    batch["boundary_target"].to(device),
                    batch["boundary_support"].to(device),
                    batch["smooth_support"].to(device),
                )
                loss = (
                    cfg["w_point"] * loss_point
                    + cfg["w_region"] * loss_region
                    + cfg["w_pseudo"] * loss_pseudo
                    + cfg["w_presence"] * loss_presence
                    + cfg["w_proto"] * loss_proto
                    + cfg["w_shadow"] * loss_shadow
                    + cfg["w_shadow_head"] * loss_shadow_head
                    + cfg["w_boundary"] * loss_boundary
                    + cfg["w_boundary_smooth"] * loss_smooth
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(model.decoder, teacher, cfg["ema_decay"])

            totals["point"] += loss_point.item()
            totals["region"] += loss_region.item()
            totals["pseudo"] += loss_pseudo.item()
            totals["presence"] += loss_presence.item()
            totals["proto"] += loss_proto.item()
            totals["shadow"] += loss_shadow.item()
            totals["shadow_head"] += loss_shadow_head.item()
            totals["boundary"] += loss_boundary.item()
            totals["smooth"] += loss_smooth.item()
            totals["accepted"] += valid.float().mean().item()

            if max_steps is not None and global_step == 1:
                print(
                    f"[DEBUG] pseudo_valid_fraction={valid.float().mean().item():.6f} "
                    f"pseudo_pixels={int(valid.sum().item())} "
                    f"pseudo_loss={loss_pseudo.item():.6f} "
                    f"shadow_loss={loss_shadow.item():.6f}"
                )

            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        n = max(1, step)
        report = {k: v / n for k, v in totals.items()}
        report.update({"epoch": epoch, "seconds": round(time.perf_counter() - started, 2)})
        print("epoch=" + str(epoch) + " " + " ".join(f"{k}={v:.4f}" for k, v in report.items()))
        with metrics_file.open("a") as fh:
            fh.write(json.dumps(report) + "\n")
        save_checkpoint(out / "last.pt", model, teacher, prototypes, epoch, cfg)
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, teacher, prototypes, epoch, cfg)

        if stop_training:
            print(f"Smoke test finished after {global_step} optimizer step.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam_fixed.json")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg, args.resume, args.max_steps)
