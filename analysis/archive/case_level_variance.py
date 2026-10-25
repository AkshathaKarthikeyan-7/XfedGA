"""
Planning estimate for the split allocation (NOT a result): case-level spread of
paired DSC differences between two existing checkpoints, measured on the OLD
FITNESS half of each server set (never the old OOD half).

Prostate: per-case volume DSC = 2*sum|P∩G| / (sum|P| + sum|G|) over the case's
non-empty slices. Breast: per-patient mean image DSC (BUS-BRA `Case` column).
Writes results/audit/case_level_variance.json with per-case values, the SD of
paired differences, and the implied 95% CI half-width (t and bootstrap) for
several test-set sizes.

The checkpoints are the existing OOD-selected *_BEST.pt (base vs xfedga). The
SD of their paired difference is only a rough proxy for the SD between methods
after seed-averaging; treat the output as an order of magnitude.

Usage (from repo root):
    python analysis/case_level_variance.py
"""
import collections
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "audit")
sys.path.insert(0, REPO)
os.chdir(REPO)

from data.breast_dataset import BreastClientDataset  # noqa: E402
from data.prostate_dataset import ProstateClientDataset  # noqa: E402
from data.server_split import get_server_split, get_prostate_server_split  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402


def load(path, device):
    m = build_residual_unet().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def sums(model, loader, device):
    inter = p_sum = g_sum = 0.0
    per_img = []
    for imgs, masks in loader:
        pred = torch.argmax(model(imgs.to(device)), dim=1).float()
        m = masks.to(device).float()
        i = (pred * m).flatten(1).sum(1)
        ps, gs = pred.flatten(1).sum(1), m.flatten(1).sum(1)
        inter += i.sum().item(); p_sum += ps.sum().item(); g_sum += gs.sum().item()
        per_img.extend(((2 * i + 1e-6) / (ps + gs + 1e-6)).tolist())
    return inter, p_sum, g_sum, per_img


def ci_table(d, sizes, rng):
    sd = float(np.std(d, ddof=1))
    out = {"n_cases_measured": len(d), "mean_diff": float(np.mean(d)), "sd_paired_diff": sd, "by_test_size": {}}
    for n in sizes:
        t_half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        boots = [np.mean(rng.choice(d, size=n, replace=True)) for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out["by_test_size"][n] = {"t_ci_half_width": t_half, "bootstrap_ci_width_half": float((hi - lo) / 2)}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    result = {}

    cfg = yaml.safe_load(open("configs/prostate_clients.yaml"))
    fit_cases, _ = get_prostate_server_split(cfg)
    a = load("results/legacy/prostate/global_model_base_fedga_prostate_BEST_OODselected.pt", device)
    b = load("results/legacy/prostate/global_model_xfedga_fedga_prostate_BEST_OODselected.pt", device)
    per_case = []
    for case in fit_cases:
        ds = ProstateClientDataset("Decathlon", cfg["server"]["path"], cases=[case])
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        row = {"case": os.path.basename(case[0]), "n_slices": len(ds)}
        for name, m in [("base", a), ("xfedga", b)]:
            i, p, g, _ = sums(m, dl, device)
            row[f"dsc_{name}"] = 2 * i / (p + g) if (p + g) > 0 else 1.0
        per_case.append(row)
    d = np.array([r["dsc_xfedga"] - r["dsc_base"] for r in per_case])
    result["prostate"] = {"split_used": "old fitness half (16 Decathlon cases)", "per_case": per_case,
                          "dsc_base_sd_across_cases": float(np.std([r["dsc_base"] for r in per_case], ddof=1)),
                          **ci_table(d, [8, 10, 12, 16], rng)}

    cfg = yaml.safe_load(open("configs/breast_clients.yaml"))
    fit_pairs, _ = get_server_split(cfg)
    meta = {r["ID"]: r["Case"] for r in csv.DictReader(open("raw data/BUS-BRA/BUSBRA/bus_data.csv"))}
    dl = DataLoader(BreastClientDataset("BUS-BRA", cfg["server"]["path"], pairs=fit_pairs), batch_size=8, shuffle=False)
    _, _, _, img_a = sums(load("results/legacy/breast/global_model_base_fedga_BEST_OODselected.pt", device), dl, device)
    _, _, _, img_b = sums(load("results/legacy/breast/global_model_xfedga_fedga_BEST_OODselected.pt", device), dl, device)
    by_pat = collections.defaultdict(list)
    for (img_path, _), da, db in zip(fit_pairs, img_a, img_b):
        by_pat[meta[os.path.basename(img_path)[:-4]]].append(db - da)
    d = np.array([np.mean(v) for v in by_pat.values()])
    result["breast"] = {"split_used": "old fitness half (938 BUS-BRA images)", "n_images": len(fit_pairs),
                        **ci_table(d, [100, 200, 400, 600], rng)}

    with open(os.path.join(OUT, "case_level_variance.json"), "w") as f:
        json.dump(result, f, indent=2)
    for task, r in result.items():
        print(task, "n_cases", r["n_cases_measured"], "sd_paired_diff", round(r["sd_paired_diff"], 4),
              {k: round(v["t_ci_half_width"], 4) for k, v in r["by_test_size"].items()})


if __name__ == "__main__":
    main()
