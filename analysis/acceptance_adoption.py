"""
Acceptance check: screen queue and adoption decision (pre-registration a689890, plan.md, 18 Oct 2026). VALIDATION ONLY.

  selected method = XFedGA+ if adopted (results/xfedga_plus/xplus_adoption.json), else XFedGA (one choice, both tasks)
  protocol        = adopted per task (results/protocol_v2/v2_adoption.json)
  acceptance runs = selected method + ga.acceptance_check=true, seeds 1-2, tag accscreen, no test
  comparator      = the selected method's seed 1-2 runs of the same protocol:
                    breast: XFedGA -> untagged v1 headline (v2 -> *_v2screen); XFedGA+ -> *_xplusscreen
                    prostate_ms: *_ms<protocol>screen
  score           = max centred 3-round mean validation DSC per run, mean of seeds 1-2
  pooled s.d.     = per task, sqrt(mean of the 2 cells' seed variances, ddof = 1)
  adopt           = beats the selected method by > pooled s.d. on >= 1 task AND not worse by > pooled s.d. on the other

    & $py analysis/acceptance_adoption.py --make_queue     # writes queues/acceptance_screen.yaml
    & $py analysis/acceptance_adoption.py                  # decision -> results/acceptance/acceptance_adoption.{json,md}
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_sets import adopted_protocols, overrides, xplus_adopted  # noqa: E402
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "acceptance")
TASKS = ["breast", "prostate_ms"]
SEEDS = [1, 2]


def selected():
    return "xfedga_plus" if xplus_adopted() else "xfedga"


def comparator_tag(task, method, proto):
    if task == "prostate_ms":
        return f"_ms{proto}screen"
    if method == "xfedga_plus":
        return "_xplusscreen"
    return "" if proto == "v1" else "_v2screen"


def make_queue():
    m, protos = selected(), adopted_protocols()
    runs = [{"task": t, "method": m, "seed": s, "set": overrides(t, protos[t]) + ["ga.acceptance_check=true"],
             "no_test": True, "tag": "accscreen"} for s in SEEDS for t in TASKS]
    q = {"name": "acceptance_screen", "out": "results/runs",
         "note": f"Acceptance-check screen (pre-registration a689890): {m} + ga.acceptance_check, protocols {protos}, "
                 "validation only, no test.", "runs": runs}
    path = os.path.join(REPO, "queues", "acceptance_screen.yaml")
    yaml.safe_dump(q, open(path, "w", encoding="utf-8"), sort_keys=False, width=200)
    print(f"{len(runs)} runs -> {path}; method {m}; protocols {protos}")


def load(task, method, tag, proto):
    pat = re.compile(rf"{task}_{method}_seed([12])_[0-9a-f]{{8}}{re.escape(tag)}")
    out = []
    for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed*")):
        mm = pat.fullmatch(os.path.basename(d))
        if not mm or not os.path.exists(os.path.join(d, "run_meta.json")) or not uses_current_cam(d):
            continue
        cfg = yaml.safe_load(open(os.path.join(d, "config_resolved.yaml"), encoding="utf-8"))
        assert (cfg["train"]["local_epochs"] == 2) == (proto == "v2"), f"{d}: protocol mismatch"
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        if tag == "_accscreen":
            assert cfg["ga"].get("acceptance_check") is True and not meta.get("test_evaluated"), d
        out.append({"seed": int(mm.group(1)), "run_id": os.path.basename(d), "score": max(meta["smoothed_val_dsc"])})
    return sorted(out, key=lambda r: r["seed"])


def decide():
    m, protos = selected(), adopted_protocols()
    res = {}
    for t in TASKS:
        cells = {}
        for name, tag in [("selected", comparator_tag(t, m, protos[t])), ("acceptance", "_accscreen")]:
            rs = load(t, m, tag, protos[t])
            s = [r["score"] for r in rs]
            cells[name] = {"runs": [r["run_id"] for r in rs], "seeds": [r["seed"] for r in rs], "scores": s,
                           "mean": float(np.mean(s)) if s else float("nan"),
                           "var": float(np.var(s, ddof=1)) if len(s) > 1 else float("nan")}
        complete = all(c["seeds"] == SEEDS for c in cells.values())
        pooled = float(np.sqrt(np.mean([c["var"] for c in cells.values()]))) if complete else float("nan")
        margin = cells["acceptance"]["mean"] - cells["selected"]["mean"]
        res[t] = {"protocol": protos[t], "cells": cells, "pooled_sd": pooled, "complete": complete,
                  "margin_acceptance_minus_selected": margin,
                  "beats": bool(complete and margin > pooled), "worse": bool(complete and margin < -pooled)}
    complete = all(r["complete"] for r in res.values())
    adopt = complete and any(r["beats"] for r in res.values()) and not any(r["worse"] for r in res.values())
    outcome = ("undecided: screen incomplete" if not complete else
               "adopt the acceptance check" if adopt else "do not adopt the acceptance check (reported as screened)")
    out = {"preregistered": "plan.md, acceptance check (commit a689890)", "selected_method": m, "decision": res,
           "adopt_acceptance": bool(adopt), "outcome": outcome}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "acceptance_adoption.json"), "w", encoding="utf-8"), indent=1)
    L = [f"| Task (protocol) | {m} | {m} + acceptance check | Margin | Pooled s.d. | Beats / worse |", "|---|---|---|---|---|---|"]
    for t, r in res.items():
        f = lambda c: f"{c['mean']:.4f} (" + ", ".join(f"{x:.4f}" for x in c["scores"]) + ")"
        L.append(f"| {t} ({r['protocol']}) | {f(r['cells']['selected'])} | {f(r['cells']['acceptance'])} | "
                 f"{r['margin_acceptance_minus_selected']:+.4f} | {r['pooled_sd']:.4f} | {r['beats']} / {r['worse']} |")
    L += ["", f"**Decision: {outcome}.** Validation only (max centred 3-round mean validation DSC, seeds 1–2)."]
    open(os.path.join(OUT, "acceptance_adoption.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_queue", action="store_true")
    a = ap.parse_args()
    make_queue() if a.make_queue else decide()
