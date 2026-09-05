"""Quick sanity check: what does raw, un-adapted SAM produce on one RS image?

Usage:
    .venv/bin/python try_sam_on_one_image.py \
        --image /agric_1962_real.png \
        --checkpoint /sam_vit_b_01ec64.pth \
        --point 128 128 \
        --label 1

--point takes pixel coords (x y) in the ORIGINAL image resolution.
--label: 1 = positive prompt ("this belongs to the object"),
         0 = negative prompt ("this does NOT belong to the object").
You can pass --point multiple times to give multiple prompts, e.g.:
    --point 128 128 --label 1 --point 10 10 --label 0
(if you extend the arg parsing below to support that — kept simple here
 with a single point for a first look).

Output: saves a 3-panel figure (original / SAM's 3 mask candidates with
scores) to ./sam_raw_output.png so you can literally see what SAM gives you.
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def show_mask(mask, ax, color):
    h, w = mask.shape
    mask_img = np.zeros((h, w, 4), dtype=np.float32)
    mask_img[mask] = color
    ax.imshow(mask_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to a raw RS image")
    parser.add_argument("--checkpoint", required=True, help="Path to sam_vit_b_01ec64.pth")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--point", nargs=2, type=int, required=True,
                         metavar=("X", "Y"), help="Point prompt in image pixel coords")
    parser.add_argument("--label", type=int, default=1, choices=[0, 1],
                         help="1=positive prompt, 0=negative prompt")
    parser.add_argument("--out", default="sam_raw_output.png")
    args = parser.parse_args()

    # --- Load image ---
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Loaded image: {args.image}, shape={image_rgb.shape}")

    # --- Load SAM ---
    print(f"Loading SAM ({args.model_type}) from {args.checkpoint} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.eval()
    predictor = SamPredictor(sam)

    # --- Set image (this runs the image encoder once) ---
    predictor.set_image(image_rgb)

    # --- Prompt SAM with a single point ---
    point_coords = np.array([args.point])       # shape (1, 2) -> [[x, y]]
    point_labels = np.array([args.label])        # shape (1,)   -> [1] or [0]

    # multimask_output=True asks SAM for its 3 candidate masks
    # (it's inherently ambiguous what "the object" means from one point,
    #  so SAM hedges by returning whole/part/subpart level candidates)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )
    print(f"SAM returned {masks.shape[0]} candidate masks.")
    for i, s in enumerate(scores):
        print(f"  Mask {i}: predicted IoU/confidence score = {s:.4f}, "
              f"pixel area = {masks[i].sum()}")

    # --- Visualize ---
    fig, axes = plt.subplots(1, masks.shape[0] + 1, figsize=(6 * (masks.shape[0] + 1), 6))

    axes[0].imshow(image_rgb)
    axes[0].scatter(*args.point, color="lime" if args.label == 1 else "red",
                     marker="*", s=300, edgecolor="white", linewidth=2)
    axes[0].set_title("Input + prompt point", fontsize=16, fontweight="bold")
    axes[0].axis("off")

    # Figure out which candidate is best
    best_idx = int(np.argmax(scores))

    for i in range(masks.shape[0]):
        axes[i + 1].imshow(image_rgb)
        show_mask(masks[i], axes[i + 1], color=np.array([1.0, 0.3, 0.1, 0.55]))
        axes[i + 1].scatter(*args.point, color="lime" if args.label == 1 else "red",
                             marker="*", s=300, edgecolor="white", linewidth=2)
        label = "BEST " if i == best_idx else ""
        area_pct = 100.0 * masks[i].sum() / (masks[i].shape[0] * masks[i].shape[1])
        axes[i + 1].set_title(
            f"{label}Candidate {i}\n"
            f"Score: {scores[i]:.3f}  |  Area: {area_pct:.1f}%",
            fontsize=14, fontweight="bold" if i == best_idx else "normal"
        )
        axes[i + 1].axis("off")

    plt.suptitle(
        f"SAM Raw Output  |  Point: ({args.point[0]}, {args.point[1]})  |  Label: {'positive' if args.label else 'negative'}",
        fontsize=18, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {args.out}")


if __name__ == "__main__":
    main()