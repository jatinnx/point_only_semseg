"""Base experiment configuration for the point-only SAM framework (v2).

Every experiment (E1..E8) is one Python file that derives from this dataclass
(usually via ``dataclasses.replace`` of the previous experiment's CONFIG), so
the diff between two experiments is exactly the flags that changed.

Central invariant: **no dense masks anywhere in the training split**. The
training manifest has no ``mask`` key (enforced by the dataset) and this
codebase contains no dense cross-entropy loss at all. Training points come
from the Chakraborty paper's own point annotations (data/make_manifests.py).

Prototype lifetime (the v2 fix): prototypes are initialised from all training
points before epoch 1, updated EVERY step from the model's own high-confidence
predictions (``proto_self_conf_threshold``), and fully refreshed every
``proto_refresh_every`` epochs — they never go stale.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> str:
    """Resolve a path so the caller's cwd never matters.

    Absolute paths pass through. Relative paths are anchored to the package
    root, EXCEPT when an identical path exists relative to the cwd. Fresh
    (not-yet-created) relative paths default to the package root.
    """
    if os.path.isabs(path):
        return path
    pkg_cand = PACKAGE_ROOT / path
    if pkg_cand.exists():
        return str(pkg_cand)
    if os.path.exists(path):
        return path
    return str(pkg_cand)


@dataclass
class Config:
    # --- identity ---
    experiment: str = "E0"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 2

    # --- data ---
    train_manifest: str = "/home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/data/train.json"      # points only, NO mask key
    val_manifest: str = "/home/cse-sdpl/Downloads/point_only_semseg/point_only_sam_rs_Es_5pt/data/val.json"          # masks present (remapped 0..16 at write time)
    image_size: int = 256
    num_classes: int = 17                        # DLRSD classes 0..16 after remap
    background_class: int | None = None          # DLRSD has no background class

    # --- active learning / multi-scale (E6) ---
    multi_scale_crop: bool = False               # random crop at 0.5-1.0 scale, resize back
    crop_scale_lo: float = 0.5                  # minimum crop scale (fraction of image)
    crop_scale_hi: float = 1.0                  # maximum crop scale (1.0 = no crop)

    # --- model ---
    sam_checkpoint: str = "/home/cse-sdpl/Downloads/point_only_semseg/sam_vit_b_01ec64.pth"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    spatial_context: bool = False        # E7: two extra 3x3 convs -> 9x9 receptive field

    # --- training ---
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    ema_decay: float = 0.999
    grad_clip: float = 5.0
    save_dir: str = "runs/checkpoints"
    save_every: int = 5

    # --- experiment ladder: which components are on ---
    use_prototypes: bool = False         # E2: live prototype bank + cosine reg (self-supervised)
    use_teacher_student: bool = False    # E3: EMA teacher, per-pixel-gated pseudo labels, consistency
    use_sam_prompt_masks: bool = False   # E3+: per-class SAM binary masks into the fusion
    use_proto_reg: bool = False          # E4: FIFO bank growth from TEACHER pseudo pixels (E2 grows from self)
    use_negative_prompts: bool = False   # E5: nearest competing-class points as negatives for SAM
    use_confidence_fusion: bool = False  # E6: 3-component confidence gate (agreement+boundary+prototype)
    use_structural_loss: bool = False    # E7: image-aware boundary smoothness loss

    # --- losses ---
    l_point: float = 1.0        # always on (the base supervision)
    l_pseudo: float = 1.0
    l_consistency: float = 1.0
    l_proto: float = 0.5
    l_smooth: float = 0.2
    pseudo_warmup_epochs: int = 5   # l_pseudo ramps 0 -> l_pseudo over these epochs (E3+)
    pseudo_warmup_cosine: bool = False  # use cosine (True) or linear (False) schedule for warmup

    # --- per-pixel pseudo-label gate (E3+) ---
    pseudo_conf_min: float = 0.70
    pseudo_conf_max: float = 0.80
    tau_ramp_epochs: int = 10
    adaptive_gate: bool = False      # E4+: gate threshold = percentile of actual teacher confidence
                                     # (fixes the fixed-threshold problem where pseudo_conf_min/max
                                     # are unreachable because teacher softmax tops out at ~0.67)
    adaptive_gate_percentile: float = 0.70  # gate out the bottom 30% of pixels by confidence
    adaptive_gate_min: float = 0.30   # floor: gate never goes below this (prevents filtering everything)
    adaptive_gate_max: float = 0.80   # ceiling: gate never goes above this

    # --- class-aware pseudo-label gate (E5) ---
    use_class_aware_gate: bool = False   # E5: per-class threshold multiplier + teacher/proto agreement
    # Per-class threshold multipliers: gate_tau for class c = gate_tau * multipliers[c]
    # Ambiguous classes (spectrally similar to others) need higher confidence.
    class_pseudo_gate_multi: dict = field(default_factory=lambda: {
        4: 1.4,   # chaparral — spectrally similar to bare soil
        8: 1.3,   # field — spectrally similar to grass/trees
        12: 1.3,  # sea — spectrally similar to water
        13: 1.3,  # sand — spectrally similar to bare soil
    })
    # For these classes, reject pseudo-labels where teacher and prototype disagree.
    class_agreement_gate: dict = field(default_factory=lambda: {
        4: True,   # chaparral — teacher errors most here
        8: True,   # field
        12: True,  # sea
        13: True,  # sand
    })

    # --- prototypes (v2: LIVE bank) ---
    proto_ema: float = 0.95
    proto_feature_patch: int = 3
    proto_pixel_confidence: float = 0.80  # min fused conf for TEACHER bank pixels (E4)
    proto_sim_threshold: float = 0.30     # min cosine sim to prototype for bank acceptance
    bank_capacity: int = 512              # per-class FIFO capacity
    # NEW in v2:
    proto_refresh_every: int = 10         # full re-init from ALL train points every N epochs
    proto_self_conf_threshold: float = 0.85  # min softmax-max conf for SELF bank updates (E2)
    proto_use_refine_at_eval: bool = False   # apply bank refine() at eval time (default OFF — it
                                             # was inert at best, corrupting at worst in v1)

    # --- fusion weights (E3+/E6) ---
    fusion_w_sem: float = 0.45
    fusion_w_sam: float = 0.25
    fusion_w_proto: float = 0.30

    # --- 3-component confidence (E6) ---
    conf_lam_agreement: float = 0.25
    conf_lam_boundary: float = 0.15
    conf_lam_proto: float = 0.60
    conf_gate_min: float = 0.70
    conf_gate_max: float = 0.80

    # --- prompts ---
    max_negative_points: int = 8
    sam_mask_threshold: float = 0.5    # sigmoid(SAM logit) must exceed this for a pixel vote

    # --- misc ---
    class_weighting: bool = True
    rare_class_factor: float = 2.0

    def __post_init__(self):
        if self.device == "cuda" and not __import__("torch").cuda.is_available():
            self.device = "cpu"


def ramp(lo: float, hi: float, epoch: int, ramp_epochs: int) -> float:
    """Linear ramp from lo to hi over ramp_epochs (clamped)."""
    if ramp_epochs <= 0:
        return hi
    t = min(1.0, epoch / ramp_epochs)
    return lo + t * (hi - lo)
