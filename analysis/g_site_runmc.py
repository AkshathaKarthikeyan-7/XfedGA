"""
Item g (pre-registration afb216e): independent-site confirmation on prostate_ms_runmc (server RUNMC, 14 test cases;
clients BIDMC, BMC, HK, I2CVB, UCL; protocol v2, S = 130 mm), seeds 1-5, test evaluated once per run.
PRIMARY of g: XFedGA+ (DSC-only) vs FedGA (baseline), test DSC, case-level paired difference (seeds averaged within
case), 95% cluster bootstrap (10,000; analysis/stats.py::compare). Supported if the CI lies above 0.
Secondary (same statistics): XFedGA+ vs FedGA; XFedGA vs FedGA. Also HD95_norm, EC, pointing game per comparison.
Writes results/site_g/g_runmc.{json,md}.
    & $py analysis/g_site_runmc.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import compare, mean_sd_over_seeds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "site_g")
SEEDS = [1, 2, 3, 4, 5]
METHODS = ["fedga", "xfedga_plus_dsc", "xfedga_plus", "xfedga"]
LABEL = {"fedga": "FedGA (baseline)", "xfedga_plus_dsc": "XFedGA+ (DSC-only)", "xfedga_plus": "XFedGA+", "xfedga": "XFedGA"}
COMPS = [("xfedga_plus_dsc", "PRIMARY of g"), ("xfedga_plus", "secondary"), ("xfedga", "secondary")]
METRICS = ["dsc", "hd95_norm", "ec", "pg"]


def rows(method):
    out = []
    for s in SEEDS:
        ds = [d for d in glob.glob(os.path.join(RUNS, f"prostate_ms_runmc_{method}_seed{s}_*_runmc"))
              if re.fullmatch(rf"prostate_ms_runmc_{method}_seed{s}_[0-9a-f]{{8}}_runmc", os.path.basename(d))]
        assert len(ds) == 1, (method, s, ds)
        meta = json.load(open(os.path.join(ds[0], "run_meta.json"), encoding="utf-8"))
        assert meta.get("test_evaluated"), ds[0]
        out.append(list(csv.DictReader(open(os.path.join(ds[0], "test_per_image.csv"), encoding="utf-8"))))
    return out


def main():
    R = {m: rows(m) for m in METHODS}
    res = {"per_method": {}, "comparisons": {}}
    for m in METHODS:
        res["per_method"][m] = {k: mean_sd_over_seeds([float(np.nanmean([float(x[k]) for x in r])) for r in R[m]])
                                for k in METRICS}
    for m, role in COMPS:
        res["comparisons"][m] = {"role": role, **{k: compare(R[m], R["fedga"], k, n_boot=10000) for k in METRICS}}
    p = res["comparisons"]["xfedga_plus_dsc"]["dsc"]
    res["primary_supported"] = bool(p["ci_low"] > 0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "g_runmc.json"), "w", encoding="utf-8"), indent=1)
    L = ["| Comparison (vs FedGA (baseline)) | Role | Test DSC: method / FedGA | Δ DSC [95% CI] | Outcome | t-interval | "
         "HD95_norm Δ (outcome) | EC Δ (outcome) | PG Δ (outcome) |", "|---|---|---|---|---|---|---|---|---|"]
    fg = res["per_method"]["fedga"]["dsc"]
    for m, role in COMPS:
        c = res["comparisons"][m]
        d = c["dsc"]
        mm = res["per_method"][m]["dsc"]
        sec = " | ".join(f"{c[k]['mean_diff']:+.4f} ({c[k]['outcome']})" for k in ["hd95_norm", "ec", "pg"])
        L.append(f"| {LABEL[m]} | {role} | {mm[0]:.4f} ± {mm[1]:.4f} / {fg[0]:.4f} ± {fg[1]:.4f} | {d['mean_diff']:+.4f} "
                 f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | **{d['outcome']}** | {d['t_outcome']} | {sec} |")
    L += ["", f"**PRIMARY of g (CI > 0): {'SUPPORTED' if res['primary_supported'] else 'NOT supported'}.** "
          f"prostate_ms_runmc (server RUNMC), {p['n_cases']} test cases, seeds 1–5, case-level 95% cluster bootstrap."]
    open(os.path.join(OUT, "g_runmc.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
