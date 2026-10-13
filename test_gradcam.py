"""
Sanity-checks the Grad-CAM pipeline end to end on your trained baseline
checkpoint. Picks a real image+mask pair, runs Grad-CAM targeting the last
decoder block, and saves a 3-panel figure: original image | ground-truth
mask | Grad-CAM overlay.

This is deliberately a standalone script, separate from xai/explainability.py,
so you can iterate on visualization/debugging quickly without touching the
library code. Once this works cleanly, the same `get_gradcam` call moves into
the GA fitness function (explanation_consistency) and the faithfulness-metric
evaluation later.

Usage:
    python test_gradcam.py --checkpoint global_model_fedavg_baseline.pt --dataset BUSI --index 5
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.residual_unet import build_residual_unet
from data.breast_dataset import BreastClientDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="global_model_fedavg_baseline.pt")
    parser.add_argument("--dataset", default="BUSI", choices=["BUSI", "BUS-BRA"])
    parser.add_argument("--path", default=None, help="dataset root; defaults per --dataset")
    parser.add_argument("--index", type=int, default=0, help="which test image to visualize")
    parser.add_argument("--out", default="gradcam_test_output.png")
    args = parser.parse_args()

    default_paths = {"BUSI": "raw data/BUSI", "BUS-BRA": "raw data/BUS-BRA"}
    data_path = args.path or default_paths[args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model + checkpoint ---
    model = build_residual_unet().to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # --- load one real image+mask pair ---
    ds = BreastClientDataset(args.dataset, data_path)
    img_t, mask_t = ds[args.index]
    input_tensor = img_t.unsqueeze(0).to(device)  # (1,1,H,W)

    # --- run Grad-CAM ---
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    target_layer = model.model[1].submodule[2][1].conv.unit0.conv
    targets = [SemanticSegmentationTarget(1, np.ones((384, 384)))]  # class 1 = lesion

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H,W) in [0,1]

    # --- also grab the model's actual prediction for reference ---
    with torch.no_grad():
        pred = torch.argmax(model(input_tensor), dim=1)[0].cpu().numpy()

    # --- visualize: image | GT mask | prediction | Grad-CAM overlay ---
    img_np = img_t.squeeze(0).numpy()
    mask_np = mask_t.numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Input image")
    axes[1].imshow(img_np, cmap="gray")
    axes[1].imshow(mask_np, cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground-truth mask")
    axes[2].imshow(img_np, cmap="gray")
    axes[2].imshow(pred, cmap="Blues", alpha=0.4)
    axes[2].set_title("Model prediction")
    axes[3].imshow(img_np, cmap="gray")
    axes[3].imshow(grayscale_cam, cmap="jet", alpha=0.5)
    axes[3].set_title("Grad-CAM")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")

    # --- quick numeric sanity check: does the CAM peak fall inside the GT mask? ---
    from xai.explainability import pointing_game_accuracy, explanation_consistency
    pg = pointing_game_accuracy(grayscale_cam, mask_np)
    ec = explanation_consistency(grayscale_cam, mask_np, threshold=0.5)
    print(f"Pointing-game hit: {bool(pg)}")
    print(f"Explanation consistency (IoU @ 0.5): {ec:.4f}")


if __name__ == "__main__":
    main()
