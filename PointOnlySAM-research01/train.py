"""Train PointOnlySAM-Semantic using point labels only.

No validation dataset is constructed in this program.  That is intentional:
dense masks remain unreachable during optimisation and are used only by the
separate evaluate.py process.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from tqdm import tqdm
except ImportError:  # keep the shared environment runnable without extras
    def tqdm(iterable, **_kwargs):
        return iterable

from pointonlysam.data import PointOnlyDataset, collate
from pointonlysam.model import PointSAMSemantic
from pointonlysam.objectives import (
    PointPrototypeBank, conservative_pseudo, geometry_boundary_loss,
    illumination_consistency, point_ce, region_ce, weighted_pseudo_ce,
)
from pointonlysam.runtime import ema_update, load_config, make_teacher, seed_everything


def save_checkpoint(path: Path, model: PointSAMSemantic, teacher: torch.nn.Module,
                    prototypes: PointPrototypeBank, epoch: int, cfg: dict) -> None:
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
    if device.type != "cuda":
        print("WARNING: SAM ViT-B at 1024px is intended for CUDA and will be very slow on CPU.")

    # This constructor enforces that cfg.train_manifest has no dense masks.
    train_set = PointOnlyDataset(cfg["train_manifest"], cfg["image_size"], training=True,
                                 geometry_cache=cfg.get("geometry_cache"))
    counts = torch.zeros(cfg["num_classes"])
    for item in train_set.items:
        for _, _, class_id in item["points"]:
            counts[int(class_id)] += 1
    class_weights = (counts.clamp_min(1).rsqrt())
    class_weights = (class_weights / class_weights.mean()).clamp(max=3.0).to(device)
    image_weights = []
    for item in train_set.items:
        ids = torch.tensor([int(p[2]) for p in item["points"]], dtype=torch.long)
        image_weights.append(float(class_weights.cpu()[ids].mean()))
    sampler = WeightedRandomSampler(image_weights, num_samples=len(image_weights), replacement=True) if cfg.get("class_balanced_sampling", False) else None
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
                        collate_fn=collate)
    model = PointSAMSemantic(cfg["sam_source"], cfg["sam_checkpoint"], cfg["num_classes"],
                             use_lora=cfg["train_sam_lora"], lora_rank=cfg["lora_rank"],
                             lora_start_layer=cfg.get("lora_start_layer", 0),
                             decoder_variant=cfg.get("decoder_variant", "rgb_fusion")).to(device)
    teacher = make_teacher(model.decoder).to(device)
    prototypes = PointPrototypeBank(cfg["num_classes"], 256, device, cfg["prototype_momentum"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch = 1
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.decoder.load_state_dict(state["decoder"])
        model.load_adapter_state(state.get("sam_lora", {}))
        teacher.load_state_dict(state["teacher_decoder"])
        prototypes.load_state_dict(state["prototypes"])
        start_epoch = int(state["epoch"]) + 1

    if cfg.get("initialize_prototypes", False) and not resume:
        # One point-only sweep gives every class an initial stable anchor before
        # pseudo-label gating starts. No dense mask is accessible here.
        init_data = PointOnlyDataset(cfg["train_manifest"], cfg["image_size"], training=False)
        init_loader = DataLoader(init_data, batch_size=1, shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate)
        model.eval()
        with torch.no_grad():
            for init_batch in init_loader:
                init_image = init_batch["weak"].to(device)
                init_points = [init_batch["points"][0].to(device)]
                prototypes.update(model.encode(init_image), init_points)
        model.train()
        print(f"Initialized prototypes from human points: {int(prototypes.seen.sum())}/{cfg['num_classes']} classes.")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    metrics_file = out / "metrics.jsonl"
    print(f"Training point-only semantic model on {device}: {len(train_set)} images; no dense masks loaded.")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = {k: 0.0 for k in ("point", "region", "pseudo", "shadow", "edge", "accepted")}
        for step, batch in enumerate(tqdm(loader, desc=f"epoch {epoch}/{cfg['epochs']}"), start=1):
            weak, strong = batch["weak"].to(device), batch["strong"].to(device)
            points = [x.to(device) for x in batch["points"]]
            # Teacher, SAM geometry, and PBR targets are all generated without
            # gradients.  They are functions of images + sparse points only.
            with torch.no_grad():
                weak_features = model.encode(weak)
                teacher_logits = teacher(weak_features, weak)
                proto_prob, _ = prototypes.probabilities(weak_features)
                if proto_prob is not None:
                    proto_prob = torch.nn.functional.interpolate(proto_prob, (cfg["image_size"], cfg["image_size"]),
                                                                  mode="bilinear", align_corners=False)
                if "region_label" in batch:
                    # The immutable region bank was produced by point-prompted
                    # SAM once. Reusing it prevents repeated prompt decoding
                    # from changing targets across epochs and cuts run time.
                    region = batch["region_label"].to(device)
                    sam = torch.full((len(points), cfg["num_classes"], cfg["image_size"], cfg["image_size"]),
                                     -10.0, device=device)
                    known = region != 255
                    class_index = region.clamp(0, cfg["num_classes"] - 1).unsqueeze(1)
                    sam.scatter_(1, class_index, torch.where(known, torch.full_like(region, 10.0), torch.full_like(region, -10.0)).unsqueeze(1).float())
                    sam_masks = [sam[i] for i in range(len(points))]
                else:
                    sam_masks = [model.prompted_geometry(weak_features[i], p, cfg["max_negative_points"])
                                 for i, p in enumerate(points)]
                labels, confidence, valid = conservative_pseudo(
                    teacher_logits, sam_masks, proto_prob, warm=epoch <= cfg["warmup_epochs"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                strong_features = model.encode(strong)
                logits = model.semantic(strong_features, strong)
                active_weights = class_weights if cfg.get("use_class_weights", False) else None
                loss_point = point_ce(logits, points, active_weights)
                if "region_label" in batch:
                    loss_region = region_ce(logits, batch["region_label"].to(device),
                                            batch["region_confidence"].to(device), active_weights)
                else:
                    loss_region = logits.sum() * 0.0
                loss_pseudo = weighted_pseudo_ce(logits, labels, confidence, valid)
                loss_shadow = illumination_consistency(logits, teacher_logits)
                loss_edge = geometry_boundary_loss(logits, sam_masks, valid)
                loss = (cfg["w_point"] * loss_point + cfg.get("w_region", 0.0) * loss_region + cfg["w_pseudo"] * loss_pseudo +
                        cfg["w_shadow"] * loss_shadow + cfg["w_boundary"] * loss_edge)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_update(model.decoder, teacher, cfg["ema_decay"])
            prototypes.update(weak_features, points)
            totals["point"] += loss_point.item()
            totals["region"] += loss_region.item()
            totals["pseudo"] += loss_pseudo.item()
            totals["shadow"] += loss_shadow.item()
            totals["edge"] += loss_edge.item()
            totals["accepted"] += valid.float().mean().item()
            if max_steps is not None and step >= max_steps:
                break
        n = max(1, step)
        report = {k: v / n for k, v in totals.items()}
        report.update({"epoch": epoch, "seconds": round(time.perf_counter() - epoch_started, 2)})
        print(f"epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in report.items()))
        with metrics_file.open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        save_checkpoint(out / "last.pt", model, teacher, prototypes, epoch, cfg)
        if epoch % cfg["save_every"] == 0:
            save_checkpoint(out / f"epoch_{epoch:03d}.pt", model, teacher, prototypes, epoch, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dlrsd_pointonly_sam.json")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, help="Smoke-test cap; omit for a full epoch.")
    parser.add_argument("--epochs", type=int, help="Override configured epochs (useful for smoke tests).")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    main(config, args.resume, args.max_steps)
