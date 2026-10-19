"""
DIAGNOSTIC (not a claim): re-evaluate the LEGACY breast FedGA and XFedGA checkpoints under the corrected evaluation.
Requested 18 Oct 2026; run only when the GPU is free. No training; no current run is changed.

Checkpoints: the FINAL-round files of results/legacy/breast/ (global_model_base_fedga.pt = FedGA (baseline),
  global_model_xfedga_fedga.pt = XFedGA). The *_BEST_OODselected.pt files (selected on test) are NOT used.
  If a final-round file is missing or does not load strictly into the current Residual U-Net: stop (written to the report).
Images: the CURRENT BUS-BRA test patients, EXCLUDING every patient with any image in the OLD fitness half (the legacy
  image-level 50/50 split, data/server_split.py, seed 42; checked to give 938 / 937 as in the audit), so the old GA
  never saw them.
Metrics (current Grad-CAM: full-res layer, predicted target; data/model code as the headline): DSC, HD95_norm,
  EC (IoU@0.5), EC_mass, pointing game.
Statistics, XFedGA - FedGA:
  corrected : patient-level paired differences, 95% cluster bootstrap (10,000), WIN / TIE / LOSE (analysis/stats.py);
  Review-2  : per-image paired tests as in the Review-2 deck (slide 7): Shapiro-Wilk on the differences (alpha 0.05)
              -> paired t-test if normal, else Wilcoxon; pointing game: exact McNemar (binomial test on discordant pairs).
Survival rule (fixed in this code before any result): a Review-2 breast claim (slide 10) SURVIVES if the corrected
  outcome matches it ("Tied" <-> TIE; "Better" <-> WIN for XFedGA, i.e. higher DSC / EC / pointing game, lower HD95).
Writes results/diagnostics/legacy_reeval.json and reports/diagnostics/legacy_reeval.md.
    & $py analysis/legacy_reeval.py
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy import stats as st

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
from review2_vs_corrected import deck_table  # noqa: E402
from stats import DIRECTION, compare  # noqa: E402

LEGACY = os.path.join(REPO, "results", "legacy", "breast")
CKPT = {"fedga": "global_model_base_fedga.pt", "xfedga": "global_model_xfedga_fedga.pt"}
METRICS = [("dsc", "DSC"), ("hd95_norm", "HD95 (normalised)"), ("ec", "EC (IoU@0.5)"), ("ec_mass", "EC_mass"),
           ("pg", "Pointing game")]
OUT_MD = os.path.join(REPO, "reports", "diagnostics", "legacy_reeval.md")
OUT_JSON = os.path.join(REPO, "results", "diagnostics", "legacy_reeval.json")


def stop(msg):
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("# Legacy re-evaluation (diagnostic): STOPPED\n\n" + msg + "\n")
    sys.exit(msg)


def review2_test(d, metric):
    d = d[~np.isnan(d)]
    if metric == "pg":
        b, c = int((d > 0).sum()), int((d < 0).sum())
        p = float(st.binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
        return {"test": "exact McNemar", "p": p, "n": int(d.size), "discordant": [b, c]}
    sw = float(st.shapiro(d).pvalue) if d.size >= 3 else float("nan")
    if sw >= 0.05:
        return {"test": "paired t-test", "p": float(st.ttest_1samp(d, 0.0).pvalue), "shapiro_p": sw, "n": int(d.size)}
    return {"test": "Wilcoxon", "p": float(st.wilcoxon(d).pvalue), "shapiro_p": sw, "n": int(d.size)}


def verdict_review2(mean_d, p, metric):
    if p >= 0.05:
        return "Tied"
    return "Better" if mean_d * DIRECTION[metric] > 0 else "Worse"


def main():
    for f in CKPT.values():
        if not os.path.exists(os.path.join(LEGACY, f)):
            stop(f"Final-round checkpoint missing: results/legacy/breast/{f}. Not evaluated.")
    from data.server_split import get_server_split
    from experiments.config import load_config
    from experiments.tasks import build_task
    from fedga.evaluation import per_image_metrics
    from models.residual_unet import build_residual_unet
    from torch.utils.data import DataLoader
    import yaml
    cfg = load_config("breast")
    legacy_cfg = yaml.safe_load(open(os.path.join(REPO, "configs", "breast_clients.yaml"), encoding="utf-8"))
    old_fit, old_ood = get_server_split(legacy_cfg)
    if (len(old_fit), len(old_ood)) != (938, 937):
        stop(f"The legacy split does not reproduce 938 / 937 (got {len(old_fit)} / {len(old_ood)}). Not evaluated.")
    meta = {r["ID"]: r["Case"] for r in csv.DictReader(open(os.path.join(cfg["server"]["path"], "BUSBRA", "bus_data.csv"),
                                                            encoding="utf-8"))}
    old_fit_patients = {meta[os.path.splitext(os.path.basename(p[0]))[0]] for p in old_fit}
    test = build_task("breast", cfg, part="server").server["test"]
    keep = [i for i, g in enumerate(test.group_ids) if g not in old_fit_patients]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit = cfg["fitness"]
    rows = {}
    for m, f in CKPT.items():
        model = build_residual_unet().to(device)
        try:
            model.load_state_dict(torch.load(os.path.join(LEGACY, f), map_location=device), strict=True)
        except Exception as e:  # noqa: BLE001
            stop(f"{f} does not load strictly into the current Residual U-Net: {e}")
        batches = DataLoader(torch.utils.data.Subset(test.dataset, keep), batch_size=1, shuffle=False)
        r = per_image_metrics(model, batches, device, fit["hd95_normalizer"], fit["cam_threshold"], with_cam=True,
                              cam_target=fit["cam_target"], cam_layer=fit["cam_layer"])
        for row, i in zip(r, keep):
            row["image_id"], row["group_id"] = test.image_ids[i], test.group_ids[i]
        rows[m] = r
    _, deck, _ = deck_table()
    res = {"n_images": len(keep), "n_patients": len({test.group_ids[i] for i in keep}),
           "excluded_patients_in_old_fitness_half": len({g for g in test.group_ids if g in old_fit_patients}),
           "test_patients_total": len(set(test.group_ids)), "metrics": {}}
    for key, _ in METRICS:
        c = compare([rows["xfedga"]], [rows["fedga"]], key, n_boot=10000)
        d = np.array([float(a[key]) - float(b[key]) for a, b in zip(rows["xfedga"], rows["fedga"])])
        t = review2_test(d, key)
        mean = {m: float(np.nanmean([float(x[key]) for x in rows[m]])) for m in rows}
        claim = deck.get(key, {}).get("interpretation")
        claim_word = claim.split()[0] if claim else None
        survives = None if claim_word is None else (
            (claim_word == "Tied" and c["outcome"] == "TIE") or (claim_word == "Better" and c["outcome"] == "WIN"))
        res["metrics"][key] = {"mean": mean, "corrected": c, "review2_style": t,
                               "review2_style_verdict": verdict_review2(float(np.nanmean(d)), t["p"], key),
                               "review2_claim": claim, "claim_survives": survives}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    L = ["# Legacy re-evaluation (DIAGNOSTIC, not a claim)", "",
         "Old breast FedGA (baseline) vs old XFedGA, **final-round** checkpoints (not the test-selected ones), on the "
         f"current BUS-BRA test patients that were not in the old fitness half: {res['n_images']} images, "
         f"{res['n_patients']} patients ({res['excluded_patients_in_old_fitness_half']} of {res['test_patients_total']} "
         "test patients excluded). Current Grad-CAM. Source: `analysis/legacy_reeval.py` → "
         "`results/diagnostics/legacy_reeval.json`.", "",
         "| Metric | Old FedGA | Old XFedGA | Δ (patient-level) [95% CI] | Corrected outcome | Per-image test (Review-2 style) | "
         "Review-2-style verdict | Review-2 claim (slide 10) | Claim survives? |", "|---|---|---|---|---|---|---|---|---|"]
    for key, label in METRICS:
        v = res["metrics"][key]
        c, t = v["corrected"], v["review2_style"]
        L.append(f"| {label} | {v['mean']['fedga']:.3f} | {v['mean']['xfedga']:.3f} | {c['mean_diff']:+.4f} "
                 f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['outcome']} | {t['test']}, p = {t['p']:.2g} | "
                 f"{v['review2_style_verdict']} | {v['review2_claim'] or '— (not on slide 10)'} | "
                 f"{'—' if v['claim_survives'] is None else ('yes' if v['claim_survives'] else 'no')} |")
    surv = [label for key, label in METRICS if res["metrics"][key]["claim_survives"]]
    gone = [label for key, label in METRICS if res["metrics"][key]["claim_survives"] is False]
    L += ["", "**Which Review-2 breast conclusions survive** (rule fixed in the script before any result: the corrected "
          "patient-level outcome must match the slide-10 claim): "
          + (f"survive: {', '.join(surv)}" if surv else "none survive") + "; "
          + (f"do not survive: {', '.join(gone)}" if gone else "none fail") + ".",
          "", "This removes only the test-set checkpoint selection (audit item 1), the fitness/test patient overlap "
          "(item 4) and the per-image statistics (item 2, by showing both). It cannot remove what is baked into the old "
          "models: BUS-UCLM malignant masks dropped in training (item 3), single unseeded runs (item 2), the old "
          "Grad-CAM layer in q_i and the EC fitness (item 6) and unequal fitness data (item 8). The per-image column "
          "shows how much the statistics method alone changes the verdict."]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("written", OUT_MD)


if __name__ == "__main__":
    main()
