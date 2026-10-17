"""
Sanity check for the CAM-definition change (17 Oct 2026): on one finished beta-screen checkpoint per task
(trained with the OLD definition), compare the old CAM (half-res layer, all-pixel target) with the new one
(full-res layer, predicted-foreground target):
  - q_i per client on that client's fixed q_i subsample (its own training images, exactly what the method uses);
  - EC (Grad-CAM-mask IoU at 0.5) and pointing game on 24 validation slices/images spread over the set.
No test data. Writes results/audit/cam_change_sanity.json.

    & $py analysis/cam_change_sanity.py
"""
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from experiments.config import load_config  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from fedga.client import client_saliency_score  # noqa: E402
from fedga.evaluation import per_image_metrics  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

DEFS = {"old (halfres, all)": ("halfres", "all"), "new (fullres, pred)": ("fullres", "pred")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for task in ["prostate", "breast"]:
        run = sorted(glob.glob(os.path.join("results", "runs", "_oldcam_definition", f"{task}_xfedga_seed1_*_betascreen_b0p0")))[0]
        cfg = load_config(task)
        model = build_residual_unet().to(device)
        model.load_state_dict(torch.load(glob.glob(os.path.join(run, "*_valselected.pt"))[0], map_location=device))
        model.eval()
        clients = [build_task(task, cfg, part=k).clients[0] for k in range(len(cfg["clients"]))]
        val = build_task(task, cfg, part="server").server["validation"]
        idx = np.linspace(0, len(val.image_ids) - 1, 24).astype(int).tolist()
        loader = DataLoader(Subset(val.dataset, idx), batch_size=8)
        rec = {"checkpoint": os.path.relpath(run, REPO), "validation_images": [val.image_ids[i] for i in idx]}
        for name, (layer, target) in DEFS.items():
            q = [client_saliency_score(model, DataLoader(Subset(c.train, c.qi_indices), batch_size=8), device,
                                       cfg["fitness"]["cam_threshold"], target, layer) for c in clients]
            rows = per_image_metrics(model, loader, device, cfg["fitness"]["hd95_normalizer"], cfg["fitness"]["cam_threshold"],
                                     with_cam=True, cam_target=target, cam_layer=layer)
            good = [r for r in rows if r["dsc"] > 0.5]
            rec[name] = {"q_i": {c.name: round(v, 4) for c, v in zip(clients, q)},
                         "val_dsc_mean": float(np.mean([r["dsc"] for r in rows])),
                         "val_ec_mean": float(np.nanmean([r["ec"] for r in rows])),
                         "val_pg_mean": float(np.nanmean([r["pg"] for r in rows])),
                         "n_dsc_gt_0.5": len(good),
                         "val_pg_on_dsc_gt_0.5": float(np.nanmean([r["pg"] for r in good])) if good else float("nan"),
                         "val_ec_on_dsc_gt_0.5": float(np.nanmean([r["ec"] for r in good])) if good else float("nan")}
        out[task] = rec
        print(task, json.dumps({k: v for k, v in rec.items() if k.startswith(("old", "new"))}, indent=1))
    os.makedirs(os.path.join("results", "audit"), exist_ok=True)
    with open(os.path.join("results", "audit", "cam_change_sanity.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
