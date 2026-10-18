"""
Root-cause verification (master plan item 8, 18 Oct 2026). VALIDATION DATA AND TRAINING LOGS ONLY (no test data).
  RC1  |alpha_i - n_i/N| per client per round (methods with q_i / s_i weighting)
  RC2  share of rounds where the GA winner's validation DSC < w_agg's (needs val_dsc_wagg)
  RC3  mean (winner - w_agg) validation DSC per round (needs val_dsc_wagg)
  RC4  agreement between each client's score (q_i, or s_i for XFedGA+) and that client model's validation DSC
       (needs client_val_dsc_i). With 2 clients a per-round Spearman rho is +/-1, so we report
       the share of rounds where the higher-scored client has the higher validation DSC (= (rho + 1) / 2), and
       the Spearman correlation, pooled over rounds and seeds, of the score difference vs the val-DSC difference.
  RC5  validation-DSC curves: peak round, peak value, round-15 value, drop (mean over seeds), per protocol.
Run groups (directory suffix): "" = v1 headline, "_v2screen", "_xplusscreen", or any other tag via --groups.
Writes results/diagnostics/root_cause.json and reports/diagnostics/root_cause_verification.md.
    & $py analysis/root_cause.py
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
METHODS = ["fedavg", "fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+",
         "xfedga_plus_dsc": "XFedGA+ (DSC-only)"}


def load(group):
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*_seed*"))):
        m = re.fullmatch(r"(prostate_ms|prostate|breast)_([a-z_]+?)_seed(\d+)_[0-9a-f]{8}" + re.escape(group),
                         os.path.basename(d))
        if not m or m.group(2) not in METHODS or not os.path.exists(os.path.join(d, "run_meta.json")) \
                or not uses_current_cam(d):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        runs.append({"task": m.group(1), "method": m.group(2), "seed": int(m.group(3)), "rows": rows,
                     "n_i": meta["n_i"], "run_id": os.path.basename(d)})
    return runs


def rc(runs, task, method):
    rs = [r for r in runs if r["task"] == task and r["method"] == method]
    if not rs:
        return None
    out = {"seeds": sorted(r["seed"] for r in rs)}
    if method != "fedavg":
        dev = []
        for r in rs:
            p = np.array(r["n_i"], dtype=float) / sum(r["n_i"])
            dev += [np.abs(np.array(json.loads(x["alpha_i"])) - p) for x in r["rows"]]
        dev = np.array(dev)
        out["RC1_mean_abs_alpha_minus_nN_per_client"] = dev.mean(0).round(5).tolist()
        out["RC1_max_abs_alpha_minus_nN"] = float(dev.max())
    if all("val_dsc_wagg" in r["rows"][0] for r in rs) and method != "fedavg":
        diff = np.array([float(x["val_dsc"]) - float(x["val_dsc_wagg"]) for r in rs for x in r["rows"]])
        out["RC2_share_rounds_winner_below_wagg"] = float(np.mean(diff < 0))
        out["RC3_mean_winner_minus_wagg_val_dsc"] = float(diff.mean())
        out["RC3_rounds"] = int(len(diff))
    score_key = "s_i" if method.startswith("xfedga_plus") else "q_i"
    if method not in ("fedavg", "fedga") and all("client_val_dsc_i" in r["rows"][0] for r in rs):
        agree, ds, dv = [], [], []
        for r in rs:
            for x in r["rows"]:
                s, v = json.loads(x[score_key]), json.loads(x["client_val_dsc_i"])
                if len(s) == 2 and s[0] != s[1] and v[0] != v[1]:
                    agree.append((s[0] > s[1]) == (v[0] > v[1]))
                    ds.append(s[0] - s[1])
                    dv.append(v[0] - v[1])
        out["RC4_score"] = score_key
        out["RC4_share_rounds_higher_score_has_higher_val_dsc"] = float(np.mean(agree)) if agree else float("nan")
        out["RC4_pooled_spearman_score_diff_vs_val_diff"] = float(spearmanr(ds, dv)[0]) if len(ds) > 2 else float("nan")
        out["RC4_rounds"] = len(agree)
    v = np.array([[float(x["val_dsc"]) for x in r["rows"]] for r in rs])
    out["RC5_peak_round_per_seed"] = (v.argmax(1) + 1).tolist()
    out["RC5_peak_val_dsc_mean"] = float(v.max(1).mean())
    out["RC5_last_round_val_dsc_mean"] = float(v[:, -1].mean())
    out["RC5_drop_peak_to_last_mean"] = float((v.max(1) - v[:, -1]).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", nargs="+", default=["", "_v2screen", "_xplusscreen", "_msv1screen", "_msv2screen"])
    args = ap.parse_args()
    res = {}
    for g in args.groups:
        runs = load(g)
        for task in ["breast", "prostate_ms", "prostate"]:   # prostate (PROMISE12 -> Decathlon): SUPERSEDED
            for m in METHODS:
                r = rc(runs, task, m)
                if r:
                    res[f"{g or 'v1_headline'}/{task}/{m}"] = r
    os.makedirs(os.path.join(REPO, "results", "diagnostics"), exist_ok=True)
    with open(os.path.join(REPO, "results", "diagnostics", "root_cause.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    fmt = lambda x, p=4: "—" if x is None or (isinstance(x, float) and x != x) else (f"{x:.{p}f}" if isinstance(x, float) else str(x))
    L = ["# Root-cause verification", "",
         "Source: `analysis/root_cause.py` → `results/diagnostics/root_cause.json`. Validation data and training logs only; "
         "no test data. A dash means the quantity was not logged for that run group.", "",
         "| Group | Task | Method | Seeds | RC1 mean abs(alpha − n/N) per client | RC2 rounds winner < w_agg (val) | "
         "RC3 mean winner − w_agg (val DSC) | RC4 higher score ⇒ higher val DSC | RC4 pooled Spearman | "
         "RC5 peak rounds | RC5 peak → last |", "|---" * 11 + "|"]
    for k, r in res.items():
        g, task, m = k.split("/")
        L.append(f"| {g} | {task} | {LABEL[m]} | {r['seeds']} | {r.get('RC1_mean_abs_alpha_minus_nN_per_client', '—')} | "
                 f"{fmt(r.get('RC2_share_rounds_winner_below_wagg'), 2)} | {fmt(r.get('RC3_mean_winner_minus_wagg_val_dsc'))} | "
                 f"{fmt(r.get('RC4_share_rounds_higher_score_has_higher_val_dsc'), 2)} | "
                 f"{fmt(r.get('RC4_pooled_spearman_score_diff_vs_val_diff'), 2)} | {r['RC5_peak_round_per_seed']} | "
                 f"{r['RC5_peak_val_dsc_mean']:.3f} → {r['RC5_last_round_val_dsc_mean']:.3f} "
                 f"({-r['RC5_drop_peak_to_last_mean']:+.3f}) |")
    L += ["", "RC6 (heterogeneity / corrupted client) is addressed by the multi-site prostate and dose-response experiments."]
    os.makedirs(os.path.join(REPO, "reports", "diagnostics"), exist_ok=True)
    with open(os.path.join(REPO, "reports", "diagnostics", "root_cause_verification.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
