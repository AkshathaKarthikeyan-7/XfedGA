"""
Audit check: which selection rule produced each root-level *_BEST.pt?

Re-evaluates every root checkpoint with the training scripts' own
evaluate_dsc on the OOD half of the server split (same seed/split code the
training scripts used), then matches the DSC against each row of the
corresponding round_history_*.csv. If a BEST checkpoint reproduces the
maximum ood_dsc row, it was selected on the OOD set.

Read-only with respect to the repo: loads checkpoints, writes one JSON next
to this script.
Usage (from repo root):
    python analysis/verify_checkpoints.py
"""
import csv
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "audit")
sys.path.insert(0, REPO)
os.chdir(REPO)

from data.breast_dataset import BreastClientDataset  # noqa: E402
from data.prostate_dataset import ProstateClientDataset  # noqa: E402
from data.server_split import get_server_split, get_prostate_server_split  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402
from experiments.train_fedga import evaluate_dsc  # noqa: E402

L = "results/legacy"  # legacy checkpoints/CSVs, moved here 15 Oct 2026 (decision Q5)
CHECKS = [
    ("breast", f"{L}/breast/global_model_fedavg_baseline", f"{L}/breast/round_history_fedavg_baseline.csv"),
    ("breast", f"{L}/breast/global_model_base_fedga", f"{L}/breast/round_history_base_fedga.csv"),
    ("breast", f"{L}/breast/global_model_xfedga_fedga", f"{L}/breast/round_history_xfedga_fedga.csv"),
    ("prostate", f"{L}/prostate/global_model_base_fedga_prostate", f"{L}/prostate/round_history_base_fedga_prostate.csv"),
    ("prostate", f"{L}/prostate/global_model_xfedga_fedga_prostate", f"{L}/prostate/round_history_xfedga_fedga_prostate.csv"),
]


def ood_loader(task):
    if task == "breast":
        cfg = yaml.safe_load(open("configs/breast_clients.yaml"))
        _, pairs = get_server_split(cfg)
        ds = BreastClientDataset(cfg["server"]["name"], cfg["server"]["path"], pairs=pairs)
    else:
        cfg = yaml.safe_load(open("configs/prostate_clients.yaml"))
        _, cases = get_prostate_server_split(cfg)
        ds = ProstateClientDataset("Decathlon", cfg["server"]["path"], cases=cases)
    return DataLoader(ds, batch_size=8, shuffle=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = {}
    out = []
    for task, stem, csv_name in CHECKS:
        if task not in loaders:
            loaders[task] = ood_loader(task)
        rows = list(csv.DictReader(open(csv_name)))
        hist = [(int(r["round"]), float(r["ood_dsc"])) for r in rows]
        best_round, best_dsc = max(hist, key=lambda x: x[1])
        for suffix in ["", "_BEST_OODselected"]:
            path = f"{stem}{suffix}.pt"
            model = build_residual_unet().to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            dsc = evaluate_dsc(model, loaders[task], device)
            nearest = min(hist, key=lambda x: abs(x[1] - dsc))
            rec = {"task": task, "checkpoint": path, "csv": csv_name, "reeval_ood_dsc": dsc,
                   "nearest_csv_round": nearest[0], "nearest_csv_dsc": nearest[1],
                   "abs_diff": abs(nearest[1] - dsc), "csv_max_round": best_round,
                   "csv_max_dsc": best_dsc, "n_rounds_in_csv": len(hist)}
            print(json.dumps(rec))
            out.append(rec)
    with open(os.path.join(OUT, "verify_checkpoints.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
