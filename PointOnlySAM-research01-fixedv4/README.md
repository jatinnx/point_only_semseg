# PointOnlySAM-Semantic v2

This is a point-only dense semantic segmentation experiment for DLRSD.  It
uses no dense train mask.  The training manifest is checked at startup and
fails if it contains a `mask` field.  Dense masks are loaded exclusively by
`evaluate.py` from the held-out validation manifest.

Architecture: frozen SAM ViT-B image features feed a 17-class semantic decoder.
Point labels create class-specific SAM geometry masks with competing-class
negative prompts (NPC).  An EMA semantic teacher, those point-seeded geometry
masks, and semantic prototypes formed only at human point locations provide
conservative partial pseudo-labels.  Strong-view synthetic shadows are used
for illumination consistency; a small loss aligns semantic boundaries with
only accepted SAM geometry boundaries.  At inference the model needs an image
only—no click and no dense label.

`v1` reached 47.26% held-out mIoU. It used sparse point loss plus an online
teacher gate and was limited by a frozen 64x64 SAM embedding. `v2` adds a
strictly point-derived static region bank, rare-class balancing, and a
high-resolution RGB fusion head. The region bank must be built once before
training; it contains only conflict-filtered SAM interiors and never accesses
dense training masks.

Run from this directory:

```bash
../.venv/bin/python build_geometry_cache.py --config configs/dlrsd_pointonly_sam_v2.json
../.venv/bin/python train.py --config configs/dlrsd_pointonly_sam_v2.json
../.venv/bin/python evaluate.py --config configs/dlrsd_pointonly_sam_v2.json --checkpoint runs/dlrsd_pointonly_sam_v2/last.pt
```

`runs/dlrsd_pointonly_sam_v2/metrics.jsonl` receives one durable record per
epoch, including point, region, pseudo, shadow, and boundary losses plus the
fraction of accepted pseudo pixels. The default freezes SAM. Once this
baseline is stable, set `train_sam_lora: true` only if GPU memory permits;
that is a controlled LoRA ablation rather than the default experiment.
