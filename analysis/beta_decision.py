"""
beta decision from the beta screen (plan.md section 0, approved 15/17 Oct 2026). VALIDATION ONLY.

Rule (pre-registered before the screen finished):
  score(run)  = max over rounds of the centred 3-round moving-average validation DSC
                (the value that selects that run's checkpoint; run_meta.json "smoothed_val_dsc")
  score(beta) = mean of score(run) over the seeds of that beta
  best beta   = argmax score(beta); betas within TIE_DSC of the best are tied, and the tie is broken
                by the lower q_i variance (round-to-round variance of q_i per client, averaged over
                clients and seeds).
  shared beta = defensible if, for each task, the other task's best beta scores within one seed-SD
                (of the task's best beta) of the task's best score.

Reads results/runs/*_betascreen_b*/; writes results/beta/beta_decision.json and beta_decision.md.
    & $py analysis/beta_decision.py
"""
import csv
import glob
import json
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "beta")
TIE_DSC = 0.002
GRID = [0.0, 0.2, 0.5, 1.0]
EXPECTED_SEEDS = 2


def load_runs():
    runs = []
    for meta_path in glob.glob(os.path.join(RUNS, "*_betascreen_b*", "run_meta.json")):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        with open(os.path.join(d, "config_resolved.yaml"), encoding="utf-8") as f:
            cfg_beta = float(yaml.safe_load(f)["aggregation"]["beta"])
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        q = np.array([json.loads(r["q_i"]) for r in rows])            # rounds x clients
        runs.append({"task": meta["task"], "seed": meta["seed"], "beta": cfg_beta, "run_id": meta["run_id"],
                     "score": max(meta["smoothed_val_dsc"]), "selected_round": meta["selected_round"],
                     "q_var": float(np.mean(np.var(q, axis=0))) if q.size else float("nan"),
                     "q_mean": q.mean(axis=0).tolist() if q.size else [],
                     "alpha_var": float(np.mean(np.var(np.array([json.loads(r["alpha_i"]) for r in rows]), axis=0))),
                     "test_evaluated": meta.get("test_evaluated")})
    return runs


def decide(runs, task):
    table = []
    for beta in GRID:
        rs = [r for r in runs if r["task"] == task and r["beta"] == beta]
        if not rs:
            continue
        s = [r["score"] for r in rs]
        table.append({"beta": beta, "n_seeds": len(rs), "seeds": sorted(r["seed"] for r in rs),
                      "score_mean": float(np.mean(s)), "score_sd": float(np.std(s, ddof=1)) if len(s) > 1 else float("nan"),
                      "q_var_mean": float(np.mean([r["q_var"] for r in rs])),
                      "alpha_var_mean": float(np.mean([r["alpha_var"] for r in rs]))})
    if not table:
        return None
    top = max(t["score_mean"] for t in table)
    tied = [t for t in table if top - t["score_mean"] <= TIE_DSC]
    best = min(tied, key=lambda t: t["q_var_mean"]) if len(tied) > 1 else tied[0]
    complete = all(t["n_seeds"] == EXPECTED_SEEDS for t in table) and len(table) == len(GRID)
    return {"table": table, "best_beta": best["beta"], "tied_with_top": [t["beta"] for t in tied],
            "tie_broken_by_q_var": len(tied) > 1, "complete": complete}


def main():
    runs = load_runs()
    assert not any(r["test_evaluated"] for r in runs), "beta screen runs must not evaluate the test set"
    dec = {t: decide(runs, t) for t in ["prostate", "breast"]}
    shared = None
    if all(dec.values()):
        ok = True
        for task, other in [("prostate", "breast"), ("breast", "prostate")]:
            tab = {t["beta"]: t for t in dec[task]["table"]}
            b, ob = dec[task]["best_beta"], dec[other]["best_beta"]
            sd = tab[b]["score_sd"]
            ok &= ob in tab and (tab[b]["score_mean"] - tab[ob]["score_mean"]) <= (sd if np.isfinite(sd) else 0)
        shared = {"defensible": bool(ok), "rule": "other task's best beta within 1 seed-SD of this task's best"}
    out = {"rule": {"score": "max centred 3-round mean validation DSC per run, mean over seeds",
                    "tie_dsc": TIE_DSC, "tie_break": "lower round-to-round q_i variance"},
           "decision": dec, "shared_beta": shared, "runs": sorted(runs, key=lambda r: (r["task"], r["beta"], r["seed"]))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "beta_decision.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lines = ["| task | beta | seeds | val score mean | seed SD | q_i variance | alpha variance |", "|---|---|---|---|---|---|---|"]
    for task, d in dec.items():
        if not d:
            continue
        for t in d["table"]:
            star = " **(selected)**" if t["beta"] == d["best_beta"] else ""
            lines.append(f"| {task} | {t['beta']}{star} | {t['seeds']} | {t['score_mean']:.4f} | {t['score_sd']:.4f} | "
                         f"{t['q_var_mean']:.2e} | {t['alpha_var_mean']:.2e} |")
    with open(os.path.join(OUT, "beta_decision.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({t: (d and {k: d[k] for k in ["best_beta", "tied_with_top", "tie_broken_by_q_var", "complete"]})
                      for t, d in dec.items()}, indent=1), "\nshared:", shared)


if __name__ == "__main__":
    main()
