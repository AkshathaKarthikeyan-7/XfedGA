"""
EC_mass on the breast v1 headline (declared evaluation of fixed, already-selected checkpoints; pre-registered:
both EC definitions are reported for every method, plan.md XFedGA+ pre-registration, 18 Oct 2026). No selection.

For FedAvg, FedGA (baseline) and XFedGA, seeds 1-5 (the untagged breast headline runs, frozen code b556334): load
the validation-selected checkpoint, compute the current Grad-CAM (full-res layer, predicted target) on EVERY test
image, and record EC (IoU@cam_threshold, as stored in test_per_image.csv `ec`) and EC_mass (share of CAM mass in
the mask). Check: the recomputed IoU EC must equal the stored `ec` (same code path) — the max difference is recorded.
Comparisons (as the headline): XFedGA vs FedGA, XFedGA vs FedAvg, FedGA vs FedAvg; patient-level paired differences,
95% cluster bootstrap (10,000), WIN / TIE / LOSE.
Writes results/xai_ec/<run_id>/ec_per_image.csv and results/xai_ec/breast_headline_ec.{json,md}. New files only.
    & $py analysis/ec_mass_headline.py [--dry_run]
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
from make_tables import load_runs  # noqa: E402
from stats import compare, mean_sd_over_seeds  # noqa: E402

OUT = os.path.join(REPO, "results", "xai_ec")
METHODS = ["fedavg", "fedga", "xfedga"]
COMPS = [("xfedga", "fedga"), ("xfedga", "fedavg"), ("fedga", "fedavg")]
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA"}


def runs():
    rs = [r for r in load_runs(os.path.join(REPO, "results", "runs", "*"), "main")
          if r["task"] == "breast" and r["method"] in METHODS]
    by = {m: sorted([r for r in rs if r["method"] == m], key=lambda r: r["seed"]) for m in METHODS}
    for m in METHODS:
        assert [r["seed"] for r in by[m]] == [1, 2, 3, 4, 5], (m, [r["seed"] for r in by[m]])
    return by


def evaluate(run, test, device):
    from experiments.config import load_config  # noqa: F401
    from models.residual_unet import build_residual_unet
    from xai.explainability import cam_layer, cam_mass_in_mask, explanation_consistency, get_gradcam
    import yaml
    cfg = yaml.safe_load(open(os.path.join(run["dir"], "config_resolved.yaml"), encoding="utf-8"))
    fit = cfg["fitness"]
    model = build_residual_unet().to(device)
    model.load_state_dict(torch.load(glob.glob(os.path.join(run["dir"], "*_valselected.pt"))[0], map_location=device))
    model.eval()
    layer = cam_layer(model, fit["cam_layer"])
    stored = {r["image_id"]: float(r["ec"]) for r in run["test"]}
    rows, maxdiff = [], 0.0
    for i in range(len(test.dataset)):
        x, m = test.dataset[i]
        cam = get_gradcam(model, layer, x[None].to(device), target_class=1, cam_target=fit["cam_target"])
        m = m.numpy()
        iou = explanation_consistency(cam, m, threshold=fit["cam_threshold"])
        iid = test.image_ids[i]
        if iid in stored and not np.isnan(stored[iid]):
            maxdiff = max(maxdiff, abs(iou - stored[iid]))
        rows.append({"image_id": iid, "group_id": test.group_ids[i], "ec_iou": iou, "ec_mass": cam_mass_in_mask(cam, m)})
    return rows, maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    by = runs()
    if args.dry_run:
        print({m: [r["run_id"] for r in by[m]] for m in METHODS})
        return
    from experiments.config import load_config
    from experiments.tasks import build_task
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test = build_task("breast", load_config("breast"), part="server").server["test"]
    per, checks = {}, {}
    for m in METHODS:
        for r in by[m]:
            path = os.path.join(OUT, r["run_id"], "ec_per_image.csv")
            if os.path.exists(path):                       # resume: never recompute a finished run
                rows = list(csv.DictReader(open(path, encoding="utf-8")))
                checks[r["run_id"]] = json.load(open(path.replace(".csv", "_check.json"), encoding="utf-8"))
            else:
                rows, maxdiff = evaluate(r, test, device)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
                checks[r["run_id"]] = {"n_images": len(rows), "max_abs_diff_ec_iou_vs_stored": maxdiff}
                json.dump(checks[r["run_id"]], open(path.replace(".csv", "_check.json"), "w", encoding="utf-8"))
                print(r["run_id"], checks[r["run_id"]], flush=True)
            per[r["run_id"]] = rows
    res = {"checks": checks, "summary": {}, "comparisons": []}
    for m in METHODS:
        for k in ("ec_iou", "ec_mass"):
            vals = [float(np.nanmean([float(x[k]) for x in per[r["run_id"]]])) for r in by[m]]
            res["summary"][f"{m}/{k}"] = mean_sd_over_seeds(vals)
    for a, b in COMPS:
        for k in ("ec_iou", "ec_mass"):
            c = compare([per[r["run_id"]] for r in by[a]], [per[r["run_id"]] for r in by[b]], k, n_boot=10000)
            res["comparisons"].append({"method": a, "comparator": b, **c})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "breast_headline_ec.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = ["# Breast v1 headline: EC (IoU@0.5) and EC_mass on every test image", "",
         "Declared evaluation of the fixed, validation-selected checkpoints (no selection). Source: "
         "`analysis/ec_mass_headline.py` → `results/xai_ec/breast_headline_ec.json`.", "",
         "| Method | EC (IoU@0.5): mean ± s.d. over seeds | EC_mass: mean ± s.d. over seeds |", "|---|---|---|"]
    for m in METHODS:
        a, b = res["summary"][f"{m}/ec_iou"], res["summary"][f"{m}/ec_mass"]
        L.append(f"| {LABEL[m]} | {a[0]:.3f} ± {a[1]:.3f} | {b[0]:.3f} ± {b[1]:.3f} |")
    L += ["", "| Comparison | Metric | Patients | Mean diff. | 95% CI (cluster bootstrap) | Outcome |", "|---|---|---|---|---|---|"]
    for c in res["comparisons"]:
        L.append(f"| {LABEL[c['method']]} vs {LABEL[c['comparator']]} | {c['metric']} | {c['n_cases']} | "
                 f"{c['mean_diff']:+.4f} | [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['outcome']} |")
    L += ["", "Check: max |recomputed IoU EC − stored test `ec`| over all runs = "
          f"{max(v['max_abs_diff_ec_iou_vs_stored'] for v in checks.values()):.2e}."]
    with open(os.path.join(OUT, "breast_headline_ec.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("written", os.path.join(OUT, "breast_headline_ec.md"))


if __name__ == "__main__":
    main()
