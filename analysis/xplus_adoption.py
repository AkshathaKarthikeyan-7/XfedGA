"""
XFedGA+ adoption decision (plan.md, pre-registration "XFedGA+", 18 Oct 2026; declared post-v1 variant). VALIDATION ONLY.

  protocol(task)  = the protocol adopted by the v2 decision (results/protocol_v2/v2_adoption.json)
  runs            = seeds 1-2 of FedGA (baseline), XFedGA, XFedGA+, XFedGA+ (DSC-only) under that protocol:
                    breast: v1 -> untagged headline runs (FedGA, XFedGA); v2 -> *_v2screen; XFedGA+ variants *_xplusscreen
                    prostate_ms: all four methods from the protocol screen group (*_msv1screen or *_msv2screen)
  score(run)      = max centred 3-round mean validation DSC (run_meta.json "smoothed_val_dsc")
  score(cell)     = mean over seeds 1-2;  pooled s.d. = per task, sqrt(mean of the 4 cells' seed variances, ddof=1)
  adopt XFedGA+   = beats XFedGA by > pooled s.d. on >= 1 task AND not worse than XFedGA by > pooled s.d. on the other
Writes results/xfedga_plus/xplus_adoption.json and .md.
    & $py analysis/xplus_adoption.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "xfedga_plus")
METHODS = ["fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]
LABEL = {"fedga": "FedGA (baseline)", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+",
         "xfedga_plus_dsc": "XFedGA+ (DSC-only)"}
SEEDS = [1, 2]


def adopted_protocols():
    d = json.load(open(os.path.join(REPO, "results", "protocol_v2", "v2_adoption.json"), encoding="utf-8"))["decision"]
    assert all(v["complete"] for v in d.values()), "v2 decision incomplete"
    return {t: ("v2" if v["adopt_v2"] else "v1") for t, v in d.items()}


def run_tag(task, method, protocol):
    if task == "prostate_ms":
        return f"_ms{protocol}screen"
    if method in ("fedga", "xfedga"):
        return "" if protocol == "v1" else "_v2screen"
    return "_xplusscreen"


def load(task, method, protocol):
    pat = re.compile(rf"{task}_{method}_seed([12])_[0-9a-f]{{8}}{run_tag(task, method, protocol)}")
    out = []
    for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed*")):
        m = pat.fullmatch(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "run_meta.json")) or not uses_current_cam(d):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        if method.startswith("xfedga_plus") or task == "prostate_ms":
            assert not meta.get("test_evaluated"), f"{d}: screen runs must not evaluate the test set"
        if protocol == "v2" or method.startswith("xfedga_plus") or task == "prostate_ms":
            import yaml
            cfg = yaml.safe_load(open(os.path.join(d, "config_resolved.yaml"), encoding="utf-8"))
            is_v2 = cfg["train"]["local_epochs"] == 2
            assert is_v2 == (protocol == "v2"), f"{d}: protocol mismatch"
        out.append({"seed": int(m.group(1)), "run_id": os.path.basename(d), "score": max(meta["smoothed_val_dsc"])})
    return sorted(out, key=lambda r: r["seed"])


def main():
    protos = adopted_protocols()
    res = {}
    for task, p in protos.items():
        cells = {}
        for m in METHODS:
            rs = load(task, m, p)
            s = [r["score"] for r in rs]
            cells[m] = {"runs": [r["run_id"] for r in rs], "seeds": [r["seed"] for r in rs], "scores": s,
                        "mean": float(np.mean(s)) if s else float("nan"),
                        "var": float(np.var(s, ddof=1)) if len(s) > 1 else float("nan")}
        complete = all(c["seeds"] == SEEDS for c in cells.values())
        pooled = float(np.sqrt(np.mean([c["var"] for c in cells.values()]))) if complete else float("nan")
        margin = cells["xfedga_plus"]["mean"] - cells["xfedga"]["mean"]
        res[task] = {"protocol": p, "cells": cells, "pooled_sd": pooled, "complete": complete,
                     "margin_xplus_minus_xfedga": margin,
                     "beats": bool(complete and margin > pooled), "worse": bool(complete and margin < -pooled)}
    complete = all(r["complete"] for r in res.values())
    adopt = complete and any(r["beats"] for r in res.values()) and not any(r["worse"] for r in res.values())
    outcome = ("undecided: screen incomplete" if not complete else
               "adopt XFedGA+" if adopt else "do not adopt XFedGA+ (reported as a screened variant)")
    out = {"rule": "adopt XFedGA+ iff it beats XFedGA by > pooled s.d. on >= 1 task and is not worse by > pooled s.d. "
                   "on the other (validation score, seeds 1-2)", "preregistered": "plan.md, XFedGA+ (18 Oct 2026)",
           "decision": res, "adopt_xfedga_plus": bool(adopt), "outcome": outcome}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "xplus_adoption.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lines = ["| Task (protocol) | Method | Val score: mean (seeds 1, 2) |", "|---|---|---|"]
    for task, r in res.items():
        for m in METHODS:
            c = r["cells"][m]
            lines.append(f"| {task} ({r['protocol']}) | {LABEL[m]} | {c['mean']:.4f} ("
                         + ", ".join(f"{x:.4f}" for x in c["scores"]) + ") |")
        lines.append(f"| {task} | XFedGA+ − XFedGA = {r['margin_xplus_minus_xfedga']:+.4f}; pooled s.d. = "
                     f"{r['pooled_sd']:.4f} | beats: {r['beats']}; worse: {r['worse']} |")
    lines += ["", f"**Decision: {outcome}.** Validation only; XFedGA+ is a declared post-v1 variant."]
    with open(os.path.join(OUT, "xplus_adoption.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
