"""
Per-image statistical significance testing between two trained prostate
models on the held-out OOD test set (Decathlon Task05_Prostate) -- same
methodology as compare_significance.py (the breast version): Shapiro-Wilk
normality check, then paired t-test if both normal, else Wilcoxon
signed-rank test, plus McNemar for the binary pointing-game outcome.

Usage:
    python compare_significance_prostate.py --model_a global_model_base_fedga_prostate_BEST.pt --model_b global_model_xfedga_fedga_prostate_BEST.pt --label_a "Base FedGA" --label_b "XFedGA"
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.prostate_dataset import ProstateClientDataset
from data.server_split import get_prostate_server_split
from models.residual_unet import build_residual_unet
from fedga.ga_optimizer import dice_score, hd95
from xai.explainability import get_gradcam, explanation_consistency, pointing_game_accuracy
import yaml


def _gradcam_target_layer(model):
    return model.model[1].submodule[2][1].conv.unit0.conv


@torch.no_grad()
def per_image_scores(model, loader, device):
    model.eval()
    dscs, hds = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = torch.argmax(model(imgs), dim=1)
        for p, m in zip(pred, masks):
            dscs.append(dice_score(p.float(), m.float()).item())
            hds.append(hd95(p.float(), m.float()))
    return np.array(dscs), np.array(hds)


def per_image_explainability_scores(model, loader, device):
    model.eval()
    target_layer = _gradcam_target_layer(model)
    ecs, pgs = [], []
    n_failed = 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1].to(device)
            mask = masks[i].numpy()
            try:
                cam = get_gradcam(model, target_layer, img, target_class=1)
                ecs.append(explanation_consistency(cam, mask, threshold=0.5))
                pgs.append(pointing_game_accuracy(cam, mask))
            except Exception:
                n_failed += 1
    if n_failed:
        print(f"  (warning: Grad-CAM failed on {n_failed} images, excluded)")
    return np.array(ecs), np.array(pgs)


def compare_continuous(scores_a, scores_b, metric_name, label_a, label_b):
    print(f"\n--- {metric_name} ---")
    print(f"{label_a}: mean={scores_a.mean():.4f}, std={scores_a.std():.4f}")
    print(f"{label_b}: mean={scores_b.mean():.4f}, std={scores_b.std():.4f}")

    _, p_a = stats.shapiro(scores_a[:min(len(scores_a), 5000)])
    _, p_b = stats.shapiro(scores_b[:min(len(scores_b), 5000)])
    both_normal = p_a > 0.05 and p_b > 0.05

    if both_normal:
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_name = "paired t-test"
    else:
        stat, p_value = stats.wilcoxon(scores_a, scores_b)
        test_name = "Wilcoxon signed-rank"

    sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
    print(f"Normality: {label_a} p={p_a:.4f}, {label_b} p={p_b:.4f} -> {'both normal' if both_normal else 'not both normal'}")
    print(f"Test used: {test_name}")
    print(f"p-value: {p_value:.6e}  -> {sig}")


def compare_binary(pgs_a, pgs_b, metric_name, label_a, label_b):
    print(f"\n--- {metric_name} ---")
    print(f"{label_a}: accuracy={pgs_a.mean():.4f} ({int(pgs_a.sum())}/{len(pgs_a)} hits)")
    print(f"{label_b}: accuracy={pgs_b.mean():.4f} ({int(pgs_b.sum())}/{len(pgs_b)} hits)")

    a_hit_b_miss = int(((pgs_a == 1) & (pgs_b == 0)).sum())
    b_hit_a_miss = int(((pgs_b == 1) & (pgs_a == 0)).sum())
    n_discordant = a_hit_b_miss + b_hit_a_miss

    if n_discordant == 0:
        print("Test used: McNemar's test")
        print("p-value: n/a -> no discordant pairs, models agree on every image")
        return

    from scipy.stats import binomtest
    result = binomtest(min(a_hit_b_miss, b_hit_a_miss), n_discordant, 0.5)
    p_value = result.pvalue
    sig = "SIGNIFICANT (p < 0.05)" if p_value < 0.05 else "not significant (p >= 0.05)"
    print(f"Discordant pairs: {label_a} hit/{label_b} miss = {a_hit_b_miss}, {label_b} hit/{label_a} miss = {b_hit_a_miss}")
    print("Test used: McNemar's test (exact binomial)")
    print(f"p-value: {p_value:.6e}  -> {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prostate_clients.yaml")
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--label_a", default="Method A")
    parser.add_argument("--label_b", default="Method B")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_explainability", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, ood_cases = get_prostate_server_split(cfg)
    ood_ds = ProstateClientDataset("Decathlon", cfg["server"]["path"], cases=ood_cases)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Evaluating on {len(ood_ds)} OOD test slices (Decathlon, from {len(ood_cases)} cases)")

    model_a = build_residual_unet().to(device)
    model_a.load_state_dict(torch.load(args.model_a, map_location=device))
    model_b = build_residual_unet().to(device)
    model_b.load_state_dict(torch.load(args.model_b, map_location=device))

    dsc_a, hd95_a = per_image_scores(model_a, ood_loader, device)
    dsc_b, hd95_b = per_image_scores(model_b, ood_loader, device)

    compare_continuous(dsc_a, dsc_b, "DSC", args.label_a, args.label_b)
    compare_continuous(hd95_a, hd95_b, "HD95 (normalized)", args.label_a, args.label_b)

    if not args.skip_explainability:
        print(f"\nRunning Grad-CAM over full OOD set for both models "
              f"({len(ood_ds)} slices each, this is slower than DSC/HD95)...")
        ec_a, pg_a = per_image_explainability_scores(model_a, ood_loader, device)
        ec_b, pg_b = per_image_explainability_scores(model_b, ood_loader, device)

        compare_continuous(ec_a, ec_b, "Explanation Consistency (EC)", args.label_a, args.label_b)
        compare_binary(pg_a, pg_b, "Pointing-Game Accuracy", args.label_a, args.label_b)


if __name__ == "__main__":
    main()
