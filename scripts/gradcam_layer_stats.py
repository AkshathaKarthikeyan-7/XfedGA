"""
Diagnostic: compare Grad-CAM target-layer activation/gradient statistics
between two checkpoints (e.g. round 2 "BEST" vs round 15 "final") on the
same OOD data, to test whether the collapsing saliency-quality scores are
caused by dead/saturated activations, vanishing gradients, or both at that
layer -- rather than the layer's *semantic* output (explanation_consistency)
just happening to score low.

Usage:
    python scripts/gradcam_layer_stats.py ^
        --ckpt_a global_model_xfedga_fedga_prostate_BEST.pt --label_a "round2_BEST" ^
        --ckpt_b global_model_xfedga_fedga_prostate.pt --label_b "round15_final" ^
        --max_samples 20
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.prostate_dataset import ProstateClientDataset
from data.server_split import get_prostate_server_split
from models.residual_unet import build_residual_unet
from experiments.train_fedga import _gradcam_target_layer


def layer_stats(model, loader, device, max_samples=20):
    """
    Runs forward + backward (target_class=1 logit sum, same as Grad-CAM's
    backward pass) on up to max_samples individual slices, capturing the
    target layer's forward activations and incoming gradients via hooks.
    Returns per-sample stats so we can see collapse trends, not just a
    single averaged number that could hide what's happening.
    """
    model.eval()
    target_layer = _gradcam_target_layer(model)

    captured = {}

    def fwd_hook(module, inp, out):
        captured["act"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        captured["grad"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    rows = []
    seen = 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            if seen >= max_samples:
                break
            img = imgs[i:i + 1].to(device).requires_grad_(False)

            model.zero_grad()
            out = model(img)  # [1, C, H, W]
            # Same target as get_gradcam(..., target_class=1): sum of the
            # foreground-class logit map, backpropagated to the target layer.
            score = out[:, 1, :, :].sum()
            score.backward()

            act = captured.get("act")
            grad = captured.get("grad")

            act_mean = act.mean().item() if act is not None else float("nan")
            act_std = act.std().item() if act is not None else float("nan")
            act_max = act.max().item() if act is not None else float("nan")
            act_nonzero_frac = (act > 0).float().mean().item() if act is not None else float("nan")

            grad_norm = grad.norm().item() if grad is not None else float("nan")
            grad_abs_mean = grad.abs().mean().item() if grad is not None else float("nan")

            rows.append({
                "act_mean": act_mean, "act_std": act_std, "act_max": act_max,
                "act_nonzero_frac": act_nonzero_frac,
                "grad_norm": grad_norm, "grad_abs_mean": grad_abs_mean,
            })
            seen += 1
        if seen >= max_samples:
            break

    h1.remove()
    h2.remove()
    return rows


def summarize(label, rows):
    print(f"\n--- {label} ({len(rows)} samples) ---")
    for key in ["act_mean", "act_std", "act_max", "act_nonzero_frac", "grad_norm", "grad_abs_mean"]:
        vals = np.array([r[key] for r in rows])
        print(f"  {key:18s}: mean={vals.mean():.6f}  std={vals.std():.6f}  "
              f"min={vals.min():.6f}  max={vals.max():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prostate_clients.yaml")
    parser.add_argument("--ckpt_a", required=True)
    parser.add_argument("--label_a", default="checkpoint_a")
    parser.add_argument("--ckpt_b", required=True)
    parser.add_argument("--label_b", default="checkpoint_b")
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, ood_cases = get_prostate_server_split(cfg)
    decathlon_root = cfg["server"]["path"]
    ood_ds = ProstateClientDataset("Decathlon", decathlon_root, cases=ood_cases)
    ood_loader = DataLoader(ood_ds, batch_size=8, shuffle=False)
    print(f"[server] Decathlon OOD-test: {len(ood_ds)} slices")

    all_rows = {}
    for ckpt_path, label in [(args.ckpt_a, args.label_a), (args.ckpt_b, args.label_b)]:
        model = build_residual_unet().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        rows = layer_stats(model, ood_loader, device, max_samples=args.max_samples)
        all_rows[label] = rows
        summarize(label, rows)

    a_rows, b_rows = all_rows[args.label_a], all_rows[args.label_b]
    print(f"\n=== {args.label_a} -> {args.label_b} ratio (mean values) ===")
    for key in ["act_mean", "act_std", "act_max", "act_nonzero_frac", "grad_norm", "grad_abs_mean"]:
        a_mean = np.mean([r[key] for r in a_rows])
        b_mean = np.mean([r[key] for r in b_rows])
        ratio = (b_mean / a_mean) if a_mean != 0 else float("inf")
        print(f"  {key:18s}: {a_mean:.6f} -> {b_mean:.6f}  (x{ratio:.3f})")


if __name__ == "__main__":
    main()
