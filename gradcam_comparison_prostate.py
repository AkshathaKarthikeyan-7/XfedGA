"""
Qualitative Grad-CAM comparison: Base FedGA vs XFedGA saliency on the same
prostate OOD test slices (Decathlon). Mirrors gradcam_comparison.py (breast
version) exactly, pointed at the prostate dataset/models instead.

Usage:
    python gradcam_comparison_prostate.py --model_a global_model_base_fedga_prostate_BEST.pt --model_b global_model_xfedga_fedga_prostate_BEST.pt --label_a "Base FedGA" --label_b "XFedGA" --n_examples 6
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.prostate_dataset import ProstateClientDataset
from data.server_split import get_prostate_server_split
from models.residual_unet import build_residual_unet
from xai.explainability import get_gradcam, explanation_consistency, pointing_game_accuracy
import yaml


def _gradcam_target_layer(model):
    return model.model[1].submodule[2][1].conv.unit0.conv


def collect_per_image(model, loader, device):
    model.eval()
    target_layer = _gradcam_target_layer(model)
    results = []
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1].to(device)
            mask = masks[i].numpy()
            try:
                cam = get_gradcam(model, target_layer, img, target_class=1)
                ec = explanation_consistency(cam, mask, threshold=0.5)
                pg = pointing_game_accuracy(cam, mask)
            except Exception:
                continue
            results.append({
                "img": img[0].detach().cpu().numpy(),
                "mask": mask,
                "cam": cam,
                "ec": ec,
                "pg": pg,
            })
    return results


def to_display_img(img_array):
    if img_array.ndim == 3:
        img_array = img_array[0]
    lo, hi = img_array.min(), img_array.max()
    return (img_array - lo) / (hi - lo + 1e-8)


def overlay_cam(base_img, cam, alpha=0.45):
    gray = to_display_img(base_img)
    heatmap = plt.cm.jet(cam)[..., :3]
    gray_rgb = np.stack([gray] * 3, axis=-1)
    return (1 - alpha) * gray_rgb + alpha * heatmap


def make_panel(example, label_a, label_b, save_path):
    img = example["img"]
    mask = example["mask"]
    cam_a, cam_b = example["cam_a"], example["cam_b"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(to_display_img(img), cmap="gray")
    axes[0].set_title("Input image")

    gray_rgb = np.stack([to_display_img(img)] * 3, axis=-1)
    gt_overlay = gray_rgb.copy()
    gt_overlay[..., 0] = np.where(mask > 0, 1.0, gt_overlay[..., 0])
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground truth (red)")

    axes[2].imshow(overlay_cam(img, cam_a))
    axes[2].set_title(f"{label_a}\nEC={example['ec_a']:.3f}, PG={'hit' if example['pg_a'] else 'miss'}")

    axes[3].imshow(overlay_cam(img, cam_b))
    axes[3].set_title(f"{label_b}\nEC={example['ec_b']:.3f}, PG={'hit' if example['pg_b'] else 'miss'}")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prostate_clients.yaml")
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--label_a", default="Method A")
    parser.add_argument("--label_b", default="Method B")
    parser.add_argument("--n_examples", type=int, default=6)
    parser.add_argument("--out_dir", default="gradcam_figures_prostate")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, ood_cases = get_prostate_server_split(cfg)
    ood_ds = ProstateClientDataset("Decathlon", cfg["server"]["path"], cases=ood_cases)
    ood_loader = DataLoader(ood_ds, batch_size=8, shuffle=False)
    print(f"Evaluating on {len(ood_ds)} OOD test slices (Decathlon, from {len(ood_cases)} cases)")

    model_a = build_residual_unet().to(device)
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b = build_residual_unet().to(device)
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    print(f"Running Grad-CAM for {args.label_a}...")
    results_a = collect_per_image(model_a, ood_loader, device)
    print(f"Running Grad-CAM for {args.label_b}...")
    results_b = collect_per_image(model_b, ood_loader, device)

    n = min(len(results_a), len(results_b))
    merged = []
    for i in range(n):
        merged.append({
            "img": results_a[i]["img"],
            "mask": results_a[i]["mask"],
            "cam_a": results_a[i]["cam"],
            "cam_b": results_b[i]["cam"],
            "ec_a": results_a[i]["ec"],
            "ec_b": results_b[i]["ec"],
            "pg_a": results_a[i]["pg"],
            "pg_b": results_b[i]["pg"],
        })

    def score(ex):
        pg_gain = ex["pg_b"] - ex["pg_a"]
        ec_gain = ex["ec_b"] - ex["ec_a"]
        return (pg_gain, ec_gain)

    merged.sort(key=score, reverse=True)

    print(f"\nSaving top {args.n_examples} examples where {args.label_b} explanation beats {args.label_a}...")
    seen_indices = set()
    saved = 0
    for idx, ex in enumerate(merged):
        if saved >= args.n_examples:
            break
        key = (ex["mask"].tobytes())
        if key in seen_indices:
            continue
        seen_indices.add(key)
        save_path = os.path.join(args.out_dir, f"comparison_{saved+1}.png")
        make_panel(ex, args.label_a, args.label_b, save_path)
        saved += 1

    print(f"\nDone. {saved} panels saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
