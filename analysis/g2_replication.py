"""
Item g2 (pre-registration c79cbc7): breast replication with NEW seeds 6-10, breast v1, test evaluated once per run.
PRIMARY of g2: XFedGA+ (DSC-only) vs FedGA (baseline), test DSC, patient-level paired difference (seeds averaged within
patient), 95% cluster bootstrap (10,000; analysis/stats.py::compare). Supported if the CI lies above 0.
Declared weaker than g: same 638 test patients as the motivating observation; only the training seeds are new.
Also reported (secondary rows, same statistics): HD95_norm, EC, pointing game.
Writes results/replication/g2_breast.{json,md}.
    & $py analysis/g2_replication.py
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
OUT = os.path.join(REPO, "results", "replication")
SEEDS = [6, 7, 8, 9, 10]


def rows(method):
    out = []
    for s in SEEDS:
        ds = [d for d in glob.glob(os.path.join(RUNS, f"breast_{method}_seed{s}_*_rep"))
              if re.fullmatch(rf"breast_{method}_seed{s}_[0-9a-f]{{8}}_rep", os.path.basename(d))]
        assert len(ds) == 1, (method, s, ds)
        meta = json.load(open(os.path.join(ds[0], "run_meta.json"), encoding="utf-8"))
        assert meta.get("test_evaluated"), ds[0]
        out.append(list(csv.DictReader(open(os.path.join(ds[0], "test_per_image.csv"), encoding="utf-8"))))
    return out


def main():
    a, b = rows("xfedga_plus_dsc"), rows("fedga")
    res = {"seeds": SEEDS, "comparisons": {}, "per_method": {}}
    for m, rr in [("xfedga_plus_dsc", a), ("fedga", b)]:
        res["per_method"][m] = {k: mean_sd_over_seeds([float(np.nanmean([float(x[k]) for x in r])) for r in rr])
                                for k in ["dsc", "hd95_norm", "ec", "pg"]}
    for k in ["dsc", "hd95_norm", "ec", "pg"]:
        res["comparisons"][k] = compare(a, b, k, n_boot=10000)
    p = res["comparisons"]["dsc"]
    res["primary_supported"] = bool(p["ci_low"] > 0)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "g2_breast.json"), "w", encoding="utf-8"), indent=1)
    L = ["| Metric | FedGA (baseline) | XFedGA+ (DSC-only) | Δ (patient-level) | 95% CI (cluster bootstrap) | Outcome | Role |",
         "|---|---|---|---|---|---|---|"]
    for k in ["dsc", "hd95_norm", "ec", "pg"]:
        c, f, x = res["comparisons"][k], res["per_method"]["fedga"][k], res["per_method"]["xfedga_plus_dsc"][k]
        L.append(f"| {k} | {f[0]:.4f} ± {f[1]:.4f} | {x[0]:.4f} ± {x[1]:.4f} | {c['mean_diff']:+.4f} | "
                 f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | {c['outcome']} | {'PRIMARY of g2' if k == 'dsc' else 'secondary'} |")
    L += ["", f"**PRIMARY of g2 (CI > 0): {'SUPPORTED' if res['primary_supported'] else 'NOT supported'}.** Breast (v1), "
          f"seeds 6–10, {p['n_cases']} test patients; replication on the same test patients (weaker than g)."]
    open(os.path.join(OUT, "g2_breast.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
