"""
Side-by-side Grad-CAM comparison between two checkpoints (e.g. round 2 BEST
vs round 15 final) on the same OOD slices -- visual follow-up to
gradcam_layer_stats.py, which showed round 15's target layer has STRONGER,
more diffuse activations/gradients than round 2, not weaker ones. This
renders the actual heatmaps so we can see whether round 15's explanation
is scattered/unfocused versus round 2's, rather than inferring it from
aggregate stats alone.

Usage:
    python scripts/gradcam_compare_checkpoints.py ^
        --ckpt_a global_model_xfedga_fedga_prostate_BEST.pt --label_a "round2_BEST" ^
        --ckpt_b global_model_xfedga_fedga_prostate.pt --label_b "round15_final" ^
        --num_slices 4
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.prostate_dataset import ProstateClientDataset
from data.server_split import get_prostate_server_split
from models.residual_unet import build_residual_unet
from experiments.train_fedga import _gradcam_target_layer
from xai.explainability import get_gradcam, explanation_consistency, pointing_game_accuracy


def load_model(ckpt_path, device):
    model = build_residual_unet().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prostate_clients.yaml")
    parser.add_argument("--ckpt_a", required=True)
    parser.add_argument("--label_a", default="checkpoint_a")
    parser.add_argument("--ckpt_b", required=True)
    parser.add_argument("--label_b", default="checkpoint_b")
    parser.add_argument("--num_slices", type=int, default=4,
                         help="Only slices with non-empty masks are chosen, "
                              "so the prostate region is actually visible.")
    parser.add_argument("--out", default="gradcam_checkpoint_comparison.png")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, ood_cases = get_prostate_server_split(cfg)
    decathlon_root = cfg["server"]["path"]
    ood_ds = ProstateClientDataset("Decathlon", decathlon_root, cases=ood_cases)
    ood_loader = DataLoader(ood_ds, batch_size=1, shuffle=False)
    print(f"[server] Decathlon OOD-test: {len(ood_ds)} slices")

    # Collect a handful of slices that actually contain prostate tissue,
    # so the comparison isn't dominated by empty-mask background slices.
    chosen = []
    for imgs, masks in ood_loader:
        mask_np = masks[0].numpy()
        if mask_np.sum() > 0:
            chosen.append((imgs, masks))
        if len(chosen) >= args.num_slices:
            break
    print(f"Selected {len(chosen)} non-empty-mask slices for comparison")

    model_a = load_model(args.ckpt_a, device)
    model_b = load_model(args.ckpt_b, device)
    target_layer_a = _gradcam_target_layer(model_a)
    target_layer_b = _gradcam_target_layer(model_b)

    n = len(chosen)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input image", "Ground-truth mask", f"Grad-CAM: {args.label_a}", f"Grad-CAM: {args.label_b}"]

    for row, (imgs, masks) in enumerate(chosen):
        img = imgs.to(device)
        mask_np = masks[0].numpy()
        img_np = imgs[0, 0].numpy()  # assumes single-channel T2 input

        cam_a = get_gradcam(model_a, target_layer_a, img, target_class=1)
        cam_b = get_gradcam(model_b, target_layer_b, img, target_class=1)

        ec_a = explanation_consistency(cam_a, mask_np, threshold=0.5)
        pg_a = pointing_game_accuracy(cam_a, mask_np)
        ec_b = explanation_consistency(cam_b, mask_np, threshold=0.5)
        pg_b = pointing_game_accuracy(cam_b, mask_np)

        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_title(col_titles[0] if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(img_np, cmap="gray")
        axes[row, 1].imshow(mask_np, cmap="Reds", alpha=0.4)
        axes[row, 1].set_title(col_titles[1] if row == 0 else "")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(img_np, cmap="gray")
        axes[row, 2].imshow(cam_a, cmap="jet", alpha=0.5)
        axes[row, 2].set_title(f"{col_titles[2]}\nEC={ec_a:.3f} PG={pg_a:.0f}" if row == 0
                                else f"EC={ec_a:.3f} PG={pg_a:.0f}")
        axes[row, 2].axis("off")

        axes[row, 3].imshow(img_np, cmap="gray")
        axes[row, 3].imshow(cam_b, cmap="jet", alpha=0.5)
        axes[row, 3].set_title(f"{col_titles[3]}\nEC={ec_b:.3f} PG={pg_b:.0f}" if row == 0
                                else f"EC={ec_b:.3f} PG={pg_b:.0f}")
        axes[row, 3].axis("off")

        print(f"  slice {row}: {args.label_a} EC={ec_a:.4f} PG={pg_a:.0f}  |  "
              f"{args.label_b} EC={ec_b:.4f} PG={pg_b:.0f}")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"\nSaved comparison figure: {args.out}")


if __name__ == "__main__":
    main()
