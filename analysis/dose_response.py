"""
Corrupted-client dose-response, tests D1 and D2 (pre-registration (a), commit 4e51538). Breast (v1), BUS-UCLM (client
index 1) corrupted at p; seeds 1-3.
  p = 0     REUSED: FedAvg / FedGA (baseline) / XFedGA = the untagged v1 breast headline seeds 1-3; XFedGA+ = the final
            breast seeds 1-3 (tag final)
  p > 0     tags dose_p025 / dose_p050 / dose_p100
D1 (primary of this experiment): per test patient, seed-averaged test DSC (patient mean over images) at each p; OLS
  slope of DSC on p per patient and method; paired per-patient difference slope(XFedGA) - slope(FedGA); 95% cluster
  bootstrap over patients (10,000; analysis/stats.py::bootstrap_ci). Supported if the CI lies above 0.
  Secondary, same test: XFedGA+ vs FedGA.
D2 (mechanism): the corrupted client's alpha, mean over rounds, per seed at each p; OLS slope over the 3 seeds x 4 p
  points; supported if the slope's 95% t-interval lies below 0. Separately for XFedGA (q_i) and XFedGA+ (s_i).
Writes results/dose_response/dose_response.{json,md}.
    & $py analysis/dose_response.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import bootstrap_ci, case_means, seed_averaged  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "dose_response")
PS = [0.0, 0.25, 0.5, 1.0]
TAG = {0.25: "_dose_p025", 0.5: "_dose_p050", 1.0: "_dose_p100"}
METHODS = ["fedavg", "fedga", "xfedga", "xfedga_plus"]
SEEDS = [1, 2, 3]
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+"}


def run_dir(method, seed, p):
    tag = TAG.get(p, "_final" if method == "xfedga_plus" else "")
    pat = re.compile(rf"breast_{method}_seed{seed}_[0-9a-f]{{8}}{re.escape(tag)}")
    ds = [d for d in glob.glob(os.path.join(RUNS, f"breast_{method}_seed{seed}_*")) if pat.fullmatch(os.path.basename(d))]
    assert len(ds) == 1, (method, seed, p, ds)
    d = ds[0]
    meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
    assert meta.get("test_evaluated"), d
    return d


def patient_dsc(d):
    return case_means(list(csv.DictReader(open(os.path.join(d, "test_per_image.csv"), encoding="utf-8"))), "dsc")


def alpha_corrupted(d):
    rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
    return float(np.mean([json.loads(r["alpha_i"])[1] for r in rows]))


def slopes(per_p):
    """per_p: {p: {patient: dsc}} -> {patient: OLS slope over the 4 p values}."""
    pats = sorted(set.intersection(*[set(v) for v in per_p.values()]))
    x = np.array(PS)
    return {c: float(np.polyfit(x, np.array([per_p[p][c] for p in PS]), 1)[0]) for c in pats}


def main():
    dirs = {(m, s, p): run_dir(m, s, p) for m in METHODS for s in SEEDS for p in PS}
    per_p = {m: {p: seed_averaged([patient_dsc(dirs[(m, s, p)]) for s in SEEDS]) for p in PS} for m in METHODS}
    sl = {m: slopes(per_p[m]) for m in METHODS}
    res = {"mean_dsc": {m: {str(p): float(np.mean(list(per_p[m][p].values()))) for p in PS} for m in METHODS},
           "mean_slope": {m: float(np.mean(list(sl[m].values()))) for m in METHODS}, "D1": {}, "D2": {}}
    for a, role in [("xfedga", "D1 primary"), ("xfedga_plus", "D1 secondary")]:
        pats = sorted(set(sl[a]) & set(sl["fedga"]))
        d = np.array([sl[a][c] - sl["fedga"][c] for c in pats])
        lo, hi = bootstrap_ci(d, n_boot=10000, seed=0)
        res["D1"][a] = {"role": role, "n_patients": len(pats), "mean_diff_slope": float(d.mean()), "ci_low": lo,
                        "ci_high": hi, "supported": bool(lo > 0)}
    for m in ["xfedga", "xfedga_plus"]:
        xs, ys = [], []
        for p in PS:
            for s in SEEDS:
                xs.append(p)
                ys.append(alpha_corrupted(dirs[(m, s, p)]))
        r = st.linregress(xs, ys)
        tcrit = st.t.ppf(0.975, len(xs) - 2)
        res["D2"][m] = {"slope": float(r.slope), "ci_low": float(r.slope - tcrit * r.stderr),
                        "ci_high": float(r.slope + tcrit * r.stderr), "n_points": len(xs),
                        "mean_alpha_by_p": {str(p): float(np.mean([y for x, y in zip(xs, ys) if x == p])) for p in PS},
                        "supported": bool(r.slope + tcrit * r.stderr < 0)}
    res["runs"] = {f"{m}/seed{s}/p{p}": os.path.basename(d) for (m, s, p), d in dirs.items()}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "dose_response.json"), "w", encoding="utf-8"), indent=1)
    L = ["| Method | " + " | ".join(f"test DSC, p = {p}" for p in PS) + " | mean slope (DSC per unit p) |",
         "|---" * (len(PS) + 2) + "|"]
    for m in METHODS:
        L.append(f"| {LABEL[m]} | " + " | ".join(f"{res['mean_dsc'][m][str(p)]:.4f}" for p in PS)
                 + f" | {res['mean_slope'][m]:+.4f} |")
    L += ["", "| Test | Comparison | Patients | Δ slope | 95% CI (cluster bootstrap) | Supported (CI > 0) |", "|---|---|---|---|---|---|"]
    for a, v in res["D1"].items():
        L.append(f"| {v['role']} | {LABEL[a]} − FedGA (baseline) | {v['n_patients']} | {v['mean_diff_slope']:+.4f} | "
                 f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {'yes' if v['supported'] else 'no'} |")
    L += ["", "| D2 (mechanism) | alpha of BUS-UCLM at p = " + ", ".join(map(str, PS)) + " | OLS slope | 95% t-interval | Supported (< 0) |",
          "|---|---|---|---|---|"]
    for m, v in res["D2"].items():
        L.append(f"| {LABEL[m]} | " + ", ".join(f"{v['mean_alpha_by_p'][str(p)]:.4f}" for p in PS)
                 + f" | {v['slope']:+.4f} | [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {'yes' if v['supported'] else 'no'} |")
    L += ["", "Breast (v1), seeds 1–3; p = 0 reused (v1 headline; XFedGA+ final runs). Test evaluated once per run. "
          "Pre-registration (a), commit 4e51538."]
    open(os.path.join(OUT, "dose_response.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
