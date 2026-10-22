"""
Decisions of the validation-only sensitivity screens h and i on prostate_ms (v2), XFedGA, seeds 1-2 (pre-registered
in plan.md, 20 Oct 2026). VALIDATION ONLY: score(run) = max centred 3-round mean validation DSC (run_meta
smoothed_val_dsc); score(cell) = mean over seeds 1-2; pooled s.d. = sqrt(mean of the cells' seed variances, ddof 1).
  h  fitness weights (w1, w2, w3): current (0.5, 0.3, 0.2) = the *_msv2screen XFedGA runs; (1,0,0) fw100,
     (0.6,0.2,0.2) fw622, (0.4,0.4,0.2) fw442. Rule: if a combination beats the current one by more than the pooled
     s.d., the highest-scoring such combination is run once with 5 seeds on test as a declared secondary;
     otherwise "insensitive".
  i  beta: 0.5 = the *_msv2screen XFedGA runs; 0, 0.2, 1.0 = ms_betascreen_b*. The pre-registered beta rule of
     b556334 (analysis/beta_decision.py::decide): keep 0.5 unless another beta beats it by more than the pooled
     s.d.; several -> the highest; ties within one pooled s.d. -> lower round-to-round q_i variance.
Also exports the breast rows of the (valid) breast beta screen for the slides.
Writes results/sensitivity/{fw_screen,beta_screen}_prostate_ms.{json,md} and reports/review/beta_breast_table.md.
    & $py analysis/sensitivity_screens.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beta_decision import decide  # noqa: E402
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "sensitivity")
FW = {"_msv2screen": (0.5, 0.3, 0.2), "_fw100": (1.0, 0.0, 0.0), "_fw622": (0.6, 0.2, 0.2), "_fw442": (0.4, 0.4, 0.2)}
BETA = {"_msv2screen": 0.5, "_ms_betascreen_b0p0": 0.0, "_ms_betascreen_b0p2": 0.2, "_ms_betascreen_b1p0": 1.0}


def load(tag):
    out = []
    for d in glob.glob(os.path.join(RUNS, "prostate_ms_xfedga_seed*")):
        m = re.fullmatch(rf"prostate_ms_xfedga_seed([12])_[0-9a-f]{{8}}{re.escape(tag)}", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "run_meta.json")) or not uses_current_cam(d):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        assert not meta.get("test_evaluated"), f"{d}: screen runs must not evaluate the test set"
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        q = np.array([json.loads(r["q_i"]) for r in rows])
        a = np.array([json.loads(r["alpha_i"]) for r in rows])
        out.append({"task": "prostate_ms", "seed": int(m.group(1)), "run_id": os.path.basename(d),
                    "score": max(meta["smoothed_val_dsc"]), "selected_round": meta["selected_round"],
                    "q_var": float(np.mean(np.var(q, axis=0))), "alpha_var": float(np.mean(np.var(a, axis=0))),
                    "test_evaluated": False})
    return sorted(out, key=lambda r: r["seed"])


def fw_screen():
    cells = {}
    for tag, w in FW.items():
        rs = load(tag)
        s = [r["score"] for r in rs]
        cells[w] = {"runs": [r["run_id"] for r in rs], "seeds": [r["seed"] for r in rs], "scores": s,
                    "mean": float(np.mean(s)) if s else float("nan"),
                    "var": float(np.var(s, ddof=1)) if len(s) > 1 else float("nan")}
    complete = all(c["seeds"] == [1, 2] for c in cells.values())
    pooled = float(np.sqrt(np.mean([c["var"] for c in cells.values()]))) if complete else float("nan")
    cur = cells[(0.5, 0.3, 0.2)]["mean"]
    beats = [w for w, c in cells.items() if w != (0.5, 0.3, 0.2) and complete and c["mean"] - cur > pooled]
    chosen = max(beats, key=lambda w: cells[w]["mean"]) if beats else None
    outcome = ("undecided: screen incomplete" if not complete else
               f"run {chosen} with 5 seeds on test (declared secondary)" if chosen else "insensitive")
    res = {"cells": {str(w): c for w, c in cells.items()}, "pooled_sd": pooled, "complete": complete,
           "beats_current": [str(w) for w in beats], "follow_up_weights": list(chosen) if chosen else None,
           "outcome": outcome}
    L = ["| (w1, w2, w3) | Val score: mean (seeds 1, 2) | − current |", "|---|---|---|"]
    for w, c in cells.items():
        L.append(f"| {w}{' (current)' if w == (0.5, 0.3, 0.2) else ''} | {c['mean']:.4f} ("
                 + ", ".join(f"{x:.4f}" for x in c["scores"]) + f") | {c['mean'] - cur:+.4f} |")
    L += ["", f"Pooled s.d. = {pooled:.4f}. **Decision: {outcome}.** prostate_ms (v2), XFedGA, validation only."]
    return res, L


def beta_screen():
    runs = []
    for tag, b in BETA.items():
        for r in load(tag):
            runs.append({**r, "beta": b})
    d = decide(runs, "prostate_ms")
    L = ["| beta | Val score: mean | seed s.d. | − beta 0.5 | beats 0.5 by > pooled s.d. | q_i variance |", "|---|---|---|---|---|---|"]
    for t in (d or {}).get("table", []):
        L.append(f"| {t['beta']}{' **(selected)**' if t['beta'] == d.get('best_beta') else ''} | {t['score_mean']:.4f} | "
                 f"{t['score_sd']:.4f} | {t.get('margin_over_default', float('nan')):+.4f} | "
                 f"{t.get('beats_default_by_more_than_pooled_sd')} | {t['q_var_mean']:.2e} |")
    L += ["", f"Pooled s.d. = {d['pooled_sd']:.4f}. **Decision ({'complete' if d['complete'] else 'INCOMPLETE'}): "
          f"{d['outcome']}.** Rule of b556334. prostate_ms (v2), XFedGA, validation only; reported as sensitivity."]
    return {"decision": d, "runs": runs}, L


def breast_beta_table():
    d = json.load(open(os.path.join(REPO, "results", "beta", "beta_decision.json"), encoding="utf-8"))["decision"]["breast"]
    L = ["# Breast beta screen (validation only; for the slides)", "",
         "Source: `results/beta/beta_decision.json` (`analysis/beta_decision.py`, rule of `b556334`; breast rows only — "
         "the prostate rows of that screen are SUPERSEDED by the PROMISE12 → Decathlon leakage).", "",
         "| beta | Val score: mean | seed s.d. | − beta 0.5 | beats 0.5 by > pooled s.d. |", "|---|---|---|---|---|"]
    for t in d["table"]:
        L.append(f"| {t['beta']}{' **(selected)**' if t['beta'] == d['best_beta'] else ''} | {t['score_mean']:.4f} | "
                 f"{t['score_sd']:.4f} | {t['margin_over_default']:+.4f} | {t['beats_default_by_more_than_pooled_sd']} |")
    L += ["", f"Pooled s.d. = {d['pooled_sd']:.4f}. Decision: {d['outcome']}. Margins are exploratory (2 seeds)."]
    return L


def fw_step0_breast():
    """Breast rows of the committed Step 0 re-ranking (results/fitness_weights/step0.json; prostate superseded)."""
    d = json.load(open(os.path.join(REPO, "results", "fitness_weights", "step0.json"), encoding="utf-8"))
    pt = d["per_task"]["breast"]
    L = ["# Fitness-weight sensitivity (w1, w2, w3), breast", "",
         "*For the panel question on the fitness weights (called lambda1-3 on the panel; w1, w2, w3 here): F = w1*DSC - "
         "w2*HD95_norm + w3*EC, default (0.5, 0.3, 0.2).*", "",
         "**Method (Step 0, offline):** the logged GA candidate pools of the 5 v1 breast-headline XFedGA runs "
         f"({pt['rounds']} rounds) are re-ranked under other weights (`analysis/fitness_weight_step0.py` -> "
         "`results/fitness_weights/step0.json`). No new training. The superseded prostate rows are excluded.", "",
         "| (w1, w2, w3) | Rounds | GA winner changes vs (0.5, 0.3, 0.2) | Winner DSC (fitness subset) | Winner HD95_norm | Winner EC |",
         "|---|---|---|---|---|---|"]
    for k, v in d["named"].items():
        if k.startswith("breast/"):
            L.append(f"| {k.split('/', 1)[1]} | {v['rounds']} | {v['n_changed']} ({v['share_winner_changed']:.0%}) | "
                     f"{v['winner_mean_dsc']:.4f} | {v['winner_mean_hd95_norm']:.4f} | {v['winner_mean_ec']:.4f} |")
    c = pt["corners"]
    L += ["", f"Over the whole 0.1-step simplex the winner changes in {pt['mean_share_changed_over_grid']:.0%} of rounds on "
          f"average; at the corners (1,0,0) {c['(1.0, 0.0, 0.0)']:.0%}, (0,1,0) {c['(0.0, 1.0, 0.0)']:.0%}, (0,0,1) "
          f"{c['(0.0, 0.0, 1.0)']:.0%}.", "",
          "**Reading:** the weights change WHICH candidate wins in many rounds, but the winner's fitness-subset DSC moves "
          "very little (table). Caveat: within-round selection only; the GA trajectory under other weights is not "
          "simulated, and winner metrics are on the fitness subset, not validation. A validation-only weight screen on "
          "`prostate_ms` (item h, plan.md) tests the trajectory effect: `results/sensitivity/fw_screen_prostate_ms.md`."]
    return L


def write_review_reports():
    os.makedirs(os.path.join(REPO, "reports", "review"), exist_ok=True)
    open(os.path.join(REPO, "reports", "review", "beta_breast_table.md"), "w", encoding="utf-8").write(
        "\n".join(breast_beta_table()) + "\n")
    open(os.path.join(REPO, "reports", "review", "fitness_weights_sensitivity.md"), "w", encoding="utf-8").write(
        "\n".join(fw_step0_breast()) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")
    for name, fn in [("fw_screen", fw_screen), ("beta_screen", beta_screen)]:
        res, L = fn()
        json.dump(res, open(os.path.join(OUT, f"{name}_prostate_ms.json"), "w", encoding="utf-8"), indent=1, default=str)
        open(os.path.join(OUT, f"{name}_prostate_ms.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        print("\n".join(L), "\n")
    write_review_reports()


if __name__ == "__main__":
    write_review_reports() if "--reports_only" in sys.argv else main()
