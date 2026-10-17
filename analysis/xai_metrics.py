"""
Independent explanation metrics from saved validation-selected checkpoints (no retraining), on the
TEST set images (the test set was already evaluated once inside the run; this script only adds
explanation metrics for the same checkpoint, it selects nothing). --split validation computes the
same metrics on validation images, e.g. to check the module on beta-screen checkpoints.

Metrics per image (spec section 6; none is optimised by XFedGA except the IoU columns, which are
reported for plausibility only):
  pg_gradcam, pg_gradcampp   pointing game (peak of the map inside the mask)
  iou_gradcampp              Grad-CAM++ mask IoU at 0.5 (plausibility, not faithfulness)
  deletion_auc, insertion_auc  mean lesion probability while the most salient pixels are removed /
                             added in 20 steps (Grad-CAM order); lower deletion / higher insertion = more faithful
Per checkpoint, on the first --randomization_images images: cascading parameter-randomisation check
(Spearman rho between the original and randomised Grad-CAM after 1, 2, 4, 8, 16 layers).

Image set: prostate = all test slices; breast = a fixed subsample of one image per test patient,
--breast_images patients (seeded by the data seed), identical for every checkpoint.

    & $py analysis/xai_metrics.py --runs "results/runs/*_seed*_*" [--limit 4]
Outputs: results/xai/<run_id>/{xai_per_image.csv, randomization.csv, summary.json}
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.splits import one_per_group_subsample  # noqa: E402
from experiments.config import load_config  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402
from xai.explainability import (explanation_consistency, get_gradcam, get_gradcam_pp, insertion_deletion_auc,  # noqa: E402
                                pointing_game_accuracy, randomization_check, unet_cam_layer)

OUT = os.path.join(REPO, "results", "xai")


def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    return out


def image_subset(task, data, cfg, breast_images, split="test"):
    test = data.server[split]
    if task == "prostate":
        return list(range(len(test.image_ids)))
    group = dict(zip(test.image_ids, test.group_ids))
    n = min(breast_images, len(set(test.group_ids)))
    return one_per_group_subsample(test.image_ids, lambda i: group[i], n, cfg["seed"])


def process(run_dir, args, device):
    meta = json.load(open(os.path.join(run_dir, "run_meta.json"), encoding="utf-8"))
    if meta.get("method_switches", {}).get("topology") == "local_only":
        return None
    if args.split == "test" and not meta.get("test_evaluated"):
        return None          # test-split metrics only for runs whose test set was already evaluated once
    out = os.path.join(args.out or OUT, meta["run_id"] + ("" if args.split == "test" else f"_{args.split}"))
    if os.path.exists(os.path.join(out, "summary.json")):
        return "skip"
    task = meta["task"]
    # the run's resolved config, with any keys added after the run was made filled from today's defaults
    cfg = _merge(load_config(task), yaml.safe_load(open(os.path.join(run_dir, "config_resolved.yaml"), encoding="utf-8")))
    data = build_task(task, cfg, part="server")
    test = data.server[args.split]
    idx = image_subset(task, data, cfg, args.breast_images, args.split)[: args.limit or None]
    ckpt = glob.glob(os.path.join(run_dir, "*_valselected.pt"))[0]
    model = build_residual_unet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    layer = unet_cam_layer(model)
    thr = cfg["fitness"]["cam_threshold"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    rows, rand_rows = [], []
    for n, i in enumerate(idx):
        img, mask = test.dataset[i]
        x, m = img[None].to(device), mask.numpy()
        cam = get_gradcam(model, layer, x, target_class=1)
        campp = get_gradcam_pp(model, layer, x, target_class=1)
        rows.append({"image_id": test.image_ids[i], "group_id": test.group_ids[i],
                     "pg_gradcam": pointing_game_accuracy(cam, m), "pg_gradcampp": pointing_game_accuracy(campp, m),
                     "iou_gradcampp": explanation_consistency(campp, m, threshold=thr),
                     "deletion_auc": insertion_deletion_auc(model, x, cam, mode="deletion"),
                     "insertion_auc": insertion_deletion_auc(model, x, cam, mode="insertion")})
        if n < args.randomization_images:
            rho = randomization_check(model, x, seed=args.seed)
            rand_rows.append({"image_id": test.image_ids[i], **{f"rho_after_{k}_layers": v for k, v in rho.items()}})
    with open(os.path.join(out, "xai_per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "randomization.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rand_rows[0]))
        w.writeheader()
        w.writerows(rand_rows)
    summary = {"run_id": meta["run_id"], "task": task, "method": meta["method"], "seed": meta["seed"], "split": args.split,
               "checkpoint": os.path.relpath(ckpt, REPO), "n_images": len(rows), "seconds": round(time.time() - t0, 1),
               **{k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("image_id", "group_id")},
               **{k: float(np.mean([r[k] for r in rand_rows])) for k in rand_rows[0] if k != "image_id"}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="glob of run folders, e.g. results/runs/*_seed*_*")
    ap.add_argument("--breast_images", type=int, default=300)
    ap.add_argument("--randomization_images", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="images per checkpoint (0 = all; for smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="validation: for checking the module on runs without a test evaluation (e.g. the beta screen)")
    ap.add_argument("--out", default=None, help="output root (default results/xai)")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_dir in sorted(glob.glob(args.runs)):
        if not os.path.exists(os.path.join(run_dir, "run_meta.json")):
            continue
        res = process(run_dir, args, device)
        if res and res != "skip":
            print(f"{res['run_id']}: {res['n_images']} images in {res['seconds']} s; "
                  f"PG {res['pg_gradcam']:.3f}, deletion {res['deletion_auc']:.3f}, insertion {res['insertion_auc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
