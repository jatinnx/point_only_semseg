# PointOnlySAM Fixed Model

This version remains strictly point-only. No dense training masks are read.
The fixed system addresses four failure modes observed in epoch-040:

1. **Class-set hallucination**: a point-supervised image-level class-presence head gates the 17-class decoder. At inference the default mode is image-only. If validation/test points are supplied, a separate `point_set` mode can hard-suppress classes that were not annotated; this is reported as an inference-time point-conditioned ablation, not the default benchmark.
2. **Prototype collapse**: four EMA prototypes are maintained per class and assigned by nearest prototype. Point-level prototype margin loss prevents a single mean vector from representing all intra-class appearances.
3. **Boundary fragmentation**: the cache stores point-derived 2-D boundary bands; the loss compares semantic neighbor disagreement in both x and y and adds an interior smoothness term. This replaces the original one-direction-only boundary loss.
4. **Shadow confusion**: the RGB branch is changed to illumination-normalized color/local-contrast/gradient cues, a learned shadow/illumination-change head gates that branch, and clean-vs-shadow semantic consistency is applied only inside known synthetic-shadow regions and only at confident teacher pixels. This is still a self-supervised illumination disentanglement mechanism, not a claimed oracle shadow detector.

## Data contract

`train_manifest` must contain only image paths and point annotations. A `mask` key in the training manifest is rejected by `FixedPointOnlyDataset`.

## Run

```bash
../.venv/bin/python build_fixed_geometry_cache.py --config configs/dlrsd_pointonly_sam_fixed.json
../.venv/bin/python train_fixed.py --config configs/dlrsd_pointonly_sam_fixed.json
../.venv/bin/python evaluate_fixed.py --config configs/dlrsd_pointonly_sam_fixed.json --checkpoint runs/dlrsd_pointonly_sam_fixed/last.pt --mode image_only
```

Optional point-set inference ablation:

```bash
../.venv/bin/python evaluate_fixed.py --config configs/dlrsd_pointonly_sam_fixed.json --checkpoint runs/dlrsd_pointonly_sam_fixed/last.pt --mode point_set
```

Image inference:

```bash
../.venv/bin/python infer_fixed.py --config configs/dlrsd_pointonly_sam_fixed.json --checkpoint runs/dlrsd_pointonly_sam_fixed/last.pt --input /path/to/images --output-dir runs/fixed_predictions
```
