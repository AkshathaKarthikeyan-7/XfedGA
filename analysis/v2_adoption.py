"""
Protocol v2 adoption decision (plan.md, pre-registration of 18 Oct 2026, commit 661a173). VALIDATION ONLY.

  score(run)      = max over rounds of the centred 3-round mean validation DSC (run_meta.json "smoothed_val_dsc")
  score(cell)     = mean of score(run) over seeds 1-2; cell = (method, protocol), method in {fedga, xfedga},
                    protocol v1 = the headline runs (no tag), v2 = the *_v2screen runs
  pooled s.d.     = per task, sqrt(mean over the four cells of the seed-to-seed variance (ddof = 1))
  decision        = adopt v2 for a task ONLY IF v2 - v1 > pooled s.d. for BOTH FedGA and XFedGA; else keep v1.
Also reported (not used in the decision): the selected rounds and, for v2, w_agg's validation DSC vs the GA
winner's (val_dsc_wagg vs val_dsc, mean over rounds) as the GA-overfitting diagnostic.

Reads results/runs; writes results/protocol_v2/v2_adoption.json and v2_adoption.md.
    & $py analysis/v2_adoption.py
"""
import csv
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
OUT = os.path.join(REPO, "results", "protocol_v2")
METHODS = ["fedga", "xfedga"]
SEEDS = [1, 2]
LABEL = {"fedga": "FedGA (baseline)", "xfedga": "XFedGA"}


def load_runs():
    runs = []
    for meta_path in glob.glob(os.path.join(RUNS, "*", "run_meta.json")):
        d = os.path.dirname(meta_path)
        name = os.path.basename(d)
        m = re.fullmatch(r"(prostate|breast)_(fedga|xfedga)_seed([12])_[0-9a-f]{8}(_v2screen)?", name)
        if not m or not uses_current_cam(d):
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        protocol = "v2" if m.group(4) else "v1"
        r = {"task": m.group(1), "method": m.group(2), "seed": int(m.group(3)), "protocol": protocol,
             "run_id": name, "score": max(meta["smoothed_val_dsc"]), "selected_round": meta["selected_round"],
             "test_evaluated": bool(meta.get("test_evaluated"))}
        if rows and "val_dsc_wagg" in rows[0]:
            r["mean_val_dsc_winner"] = float(np.mean([float(x["val_dsc"]) for x in rows]))
            r["mean_val_dsc_wagg"] = float(np.mean([float(x["val_dsc_wagg"]) for x in rows]))
            r["share_rounds_winner_below_wagg"] = float(np.mean([float(x["val_dsc"]) < float(x["val_dsc_wagg"])
                                                                for x in rows]))
        runs.append(r)
    return runs


def decide(runs, task):
    cells = {}
    for m in METHODS:
        for p in ["v1", "v2"]:
            rs = sorted([r for r in runs if r["task"] == task and r["method"] == m and r["protocol"] == p],
                        key=lambda r: r["seed"])
            s = [r["score"] for r in rs]
            cells[f"{m}/{p}"] = {"seeds": [r["seed"] for r in rs], "scores": s,
                                 "mean": float(np.mean(s)) if s else float("nan"),
                                 "var": float(np.var(s, ddof=1)) if len(s) > 1 else float("nan")}
    complete = all(c["seeds"] == SEEDS for c in cells.values())
    pooled = float(np.sqrt(np.mean([c["var"] for c in cells.values()]))) if complete else float("nan")
    margins = {m: cells[f"{m}/v2"]["mean"] - cells[f"{m}/v1"]["mean"] for m in METHODS}
    beats = {m: bool(complete and margins[m] > pooled) for m in METHODS}
    if not complete:
        outcome = "undecided: screen incomplete"
    elif all(beats.values()):
        outcome = "adopt v2: beats v1 by more than the pooled s.d. for both FedGA and XFedGA"
    else:
        outcome = "keep v1: v2 does not beat v1 by more than the pooled s.d. for " + \
                  " and ".join(LABEL[m] for m in METHODS if not beats[m])
    return {"cells": cells, "pooled_sd": pooled, "margin_v2_minus_v1": margins,
            "beats_by_more_than_pooled_sd": beats, "adopt_v2": bool(complete and all(beats.values())),
            "complete": complete, "outcome": outcome}


def main():
    runs = load_runs()
    assert not any(r["test_evaluated"] for r in runs if r["protocol"] == "v2"), "v2 screen runs must not evaluate test"
    dec = {t: decide(runs, t) for t in ["prostate", "breast"]}
    out = {"rule": "adopt v2 for a task only if v2 beats v1 (mean of seeds 1-2 of the max centred 3-round mean "
                   "validation DSC) by more than the task's pooled seed s.d. (4 cells) for BOTH FedGA and XFedGA",
           "preregistered": "plan.md, 18 Oct 2026, commit 661a173", "decision": dec,
           "runs": sorted(runs, key=lambda r: (r["task"], r["method"], r["protocol"], r["seed"]))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "v2_adoption.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lines = ["| Task | Method | v1 score (seeds 1, 2) | v2 score (seeds 1, 2) | v2 − v1 | Beats by > pooled s.d. |",
             "|---|---|---|---|---|---|"]
    for task, d in dec.items():
        for m in METHODS:
            c1, c2 = d["cells"][f"{m}/v1"], d["cells"][f"{m}/v2"]
            fmt = lambda c: f"{c['mean']:.4f} ({', '.join(f'{x:.4f}' for x in c['scores'])})"
            lines.append(f"| {task} | {LABEL[m]} | {fmt(c1)} | {fmt(c2)} | {d['margin_v2_minus_v1'][m]:+.4f} | "
                         f"{d['beats_by_more_than_pooled_sd'][m]} |")
        lines.append(f"| {task} | pooled s.d. = {d['pooled_sd']:.4f} | | | | **{d['outcome']}** |")
    lines += ["", "Validation only. Score = max centred 3-round mean validation DSC per run, mean of seeds 1–2; each "
              "protocol is measured in its own pipeline (prostate v2 crop cuts no fitness/validation gland pixels: "
              "results/audit/fov_crop.json).", "",
              "| v2 run | Mean val DSC: GA winner | Mean val DSC: w_agg | Rounds winner < w_agg |", "|---|---|---|---|"]
    for r in out["runs"]:
        if "mean_val_dsc_wagg" in r:
            lines.append(f"| {r['run_id']} | {r['mean_val_dsc_winner']:.4f} | {r['mean_val_dsc_wagg']:.4f} | "
                         f"{r['share_rounds_winner_below_wagg']:.0%} |")
    with open(os.path.join(OUT, "v2_adoption.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
