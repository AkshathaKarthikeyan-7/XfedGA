"""
Diagnostics v1 on the headline runs (18 Oct 2026). VALIDATION, TRAINING LOGS and CLIENT / SERVER
fitness+validation masks only; no test data; no training (item 4 runs inference with the saved
validation-selected checkpoints on validation slices).

  1. GA overfitting: per round, the GA winner's fitness gain over w_agg on the fitness subset
     (ga_candidates.jsonl: w_agg is the first evaluated candidate); validation DSC of w_agg is not logged.
  2. Early peak: per-round validation DSC; peak round; drop to round 15; client update norms per round
     (proxy for drift; client training loss is not logged).
  3. Prostate gland fraction of each 384x384 slice (clients; server fitness + validation sets).
  4. Prostate validation error analysis: DSC with and without largest-connected-component post-processing;
     share of predicted foreground in components that do not touch the ground truth.
Writes results/diagnostics/v1.json.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from analysis.run_filter import uses_current_cam  # noqa: E402
from experiments.config import load_config  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

HEADLINE = ["fedavg", "fedga", "xfedga"]


def headline_runs():
    runs = []
    for d in sorted(glob.glob(os.path.join("results", "runs", "*_seed[1-5]_????????"))):
        if not (os.path.exists(os.path.join(d, "run_meta.json")) and uses_current_cam(d)):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        if meta["method"] in HEADLINE:
            meta["dir"] = d
            meta["rounds"] = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
            runs.append(meta)
    return runs


def ga_gains(runs):
    out = {}
    for task in ["prostate", "breast"]:
        for m in ["fedga", "xfedga"]:
            gains, frac_kept = [], []
            for r in [r for r in runs if r["task"] == task and r["method"] == m]:
                for line in open(os.path.join(r["dir"], "ga_candidates.jsonl"), encoding="utf-8"):
                    c = json.loads(line)["candidates"]
                    f = [x["fitness"] for x in c]
                    gains.append(max(f) - f[0])          # winner (max fitness: elitism) minus w_agg (first evaluated)
                    frac_kept.append(float(np.argmax(f) == 0))
            g = np.array(gains)
            out[f"{task}/{m}"] = {"rounds": len(g), "mean_fitness_gain": float(g.mean()), "median": float(np.median(g)),
                                  "p90": float(np.quantile(g, 0.9)), "share_rounds_w_agg_kept": float(np.mean(frac_kept))}
    return out


def curves(runs):
    out = {}
    for task in ["prostate", "breast"]:
        for m in HEADLINE:
            rs = [r for r in runs if r["task"] == task and r["method"] == m]
            v = np.array([[float(x["val_dsc"]) for x in r["rounds"]] for r in rs])
            peaks = v.argmax(1) + 1
            norms = np.array([[np.mean(json.loads(x["update_norm_i"])) for x in r["rounds"]] for r in rs]) \
                if rs and "update_norm_i" in rs[0]["rounds"][0] else None
            out[f"{task}/{m}"] = {
                "mean_val_dsc_per_round": v.mean(0).round(4).tolist(), "peak_round_per_seed": peaks.tolist(),
                "peak_val_dsc_mean": float(v.max(1).mean()), "round15_val_dsc_mean": float(v[:, -1].mean()),
                "drop_peak_to_round15_mean": float((v.max(1) - v[:, -1]).mean()),
                "selected_round_per_seed": [r["selected_round"] for r in rs],
                "update_norm_mean_per_round": norms.mean(0).round(3).tolist() if norms is not None else None}
    return out


def gland_fraction():
    cfg = load_config("prostate")
    res = {}
    full = build_task("prostate", cfg)
    for c in full.clients:
        f = [float(c.train[i][1].float().mean()) for i in range(len(c.train))]
        res[c.name] = {"slices": len(f), "mean": float(np.mean(f)), "median": float(np.median(f))}
    for role in ["fitness", "validation"]:          # the test set is not read
        ds = full.server[role].dataset
        f = [float(ds[i][1].float().mean()) for i in range(len(ds))]
        res[f"Decathlon_{role}"] = {"slices": len(f), "mean": float(np.mean(f)), "median": float(np.median(f))}
    return res


def lcc(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return (lab == (int(np.argmax(sizes)) + 1)).astype(mask.dtype)


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p * g).sum() / s)


def prostate_validation_errors(runs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = build_task("prostate", load_config("prostate"), part="server").server["validation"]
    imgs = torch.stack([val.dataset[i][0] for i in range(len(val.dataset))])
    gts = [val.dataset[i][1].numpy() for i in range(len(val.dataset))]
    out = {}
    for m in HEADLINE:
        raw, post, fp_detached, n_comp = [], [], [], []
        for r in [r for r in runs if r["task"] == "prostate" and r["method"] == m]:
            model = build_residual_unet().to(device)
            model.load_state_dict(torch.load(glob.glob(os.path.join(r["dir"], "*_valselected.pt"))[0], map_location=device))
            model.eval()
            with torch.no_grad():
                preds = torch.cat([torch.argmax(model(imgs[i:i + 16].to(device)), 1).cpu() for i in range(0, len(imgs), 16)])
            for p, g in zip(preds.numpy().astype(np.uint8), gts):
                raw.append(dice(p, g))
                post.append(dice(lcc(p), g))
                lab, n = ndimage.label(p)
                n_comp.append(n)
                if p.sum():
                    touching = set(np.unique(lab[(g > 0) & (lab > 0)]))
                    detached = sum(int((lab == k).sum()) for k in range(1, n + 1) if k not in touching)
                    fp_detached.append(detached / p.sum())
        out[m] = {"val_slices_x_seeds": len(raw), "mean_dsc_raw": float(np.mean(raw)),
                  "mean_dsc_largest_component": float(np.mean(post)),
                  "delta_lcc": float(np.mean(post) - np.mean(raw)),
                  "share_slices_lcc_improves": float(np.mean(np.array(post) > np.array(raw) + 1e-9)),
                  "share_slices_lcc_hurts": float(np.mean(np.array(post) < np.array(raw) - 1e-9)),
                  "mean_components_per_slice": float(np.mean(n_comp)),
                  "mean_share_pred_fg_detached_from_gt": float(np.mean(fp_detached)) if fp_detached else float("nan")}
    return out


def main():
    runs = headline_runs()
    assert len(runs) == 30, len(runs)
    res = {"n_runs": len(runs), "ga_fitness_gain": ga_gains(runs), "curves": curves(runs),
           "gland_fraction": gland_fraction(), "prostate_validation_errors": prostate_validation_errors(runs),
           "not_logged": ["validation DSC of w_agg (only the GA winner is validated each round)",
                          "client training loss per epoch / round"]}
    os.makedirs(os.path.join("results", "diagnostics"), exist_ok=True)
    with open(os.path.join("results", "diagnostics", "v1.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "curves"}, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ["peak_round_per_seed", "peak_val_dsc_mean", "round15_val_dsc_mean",
                                                 "drop_peak_to_round15_mean", "selected_round_per_seed"]}
                      for k, v in res["curves"].items()}, indent=1))
    print("update norms per round:", {k: v["update_norm_mean_per_round"] for k, v in res["curves"].items()})


if __name__ == "__main__":
    main()
