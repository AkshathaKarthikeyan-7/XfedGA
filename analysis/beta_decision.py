"""
beta decision from the beta screen (plan.md section 0, approved 15/17 Oct 2026). VALIDATION ONLY.

Rule (revised 17 Oct 2026, 12:50, BEFORE any decision was computed: 1 of 16 screen runs finished,
none analysed; the earlier 0.002-DSC tie rule is superseded -- see git history of this file):
  score(run)   = max over rounds of the centred 3-round moving-average validation DSC
                 (the value that selects that run's checkpoint; run_meta.json "smoothed_val_dsc")
  score(beta)  = mean of score(run) over the 2 seeds of that beta
  pooled s.d.  = per task, sqrt(mean over beta of the seed-to-seed variance of score(run))
                 (variance with ddof = 1; every beta has the same number of seeds)
  decision     = keep the pre-registered default beta = 0.5 UNLESS another beta beats it by more
                 than the pooled s.d.; if several do, take the highest-scoring one; betas within one
                 pooled s.d. of that highest one are tied, and the tie is broken by the lower
                 round-to-round q_i variance (per client, averaged over clients and seeds).
  shared beta  = reported (not used): defensible if each task's selected beta lies within one pooled
                 s.d. of the other task's selected score.

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
DEFAULT_BETA = 0.5
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
    variances = [t["score_sd"] ** 2 for t in table if np.isfinite(t["score_sd"])]
    pooled_sd = float(np.sqrt(np.mean(variances))) if variances else float("nan")
    by_beta = {t["beta"]: t for t in table}
    default = by_beta.get(DEFAULT_BETA)
    complete = all(t["n_seeds"] == EXPECTED_SEEDS for t in table) and len(table) == len(GRID)
    if default is None or not np.isfinite(pooled_sd):
        return {"table": table, "pooled_sd": pooled_sd, "best_beta": None, "complete": complete,
                "outcome": "undecided: default beta or seed s.d. missing"}
    beats = [t for t in table if t["beta"] != DEFAULT_BETA and t["score_mean"] - default["score_mean"] > pooled_sd]
    if not beats:
        best, tied, outcome = default, [], "default beta 0.5 kept: no beta beats it by more than the pooled s.d."
    else:
        top = max(t["score_mean"] for t in beats)
        tied = [t for t in beats if top - t["score_mean"] <= pooled_sd]
        best = min(tied, key=lambda t: t["q_var_mean"])
        outcome = (f"beta {best['beta']} beats the default by more than the pooled s.d."
                   + (f"; tied with {[t['beta'] for t in tied if t is not best]} within one s.d., "
                      "broken by lower q_i variance" if len(tied) > 1 else ""))
    for t in table:
        t["margin_over_default"] = t["score_mean"] - default["score_mean"]
        t["beats_default_by_more_than_pooled_sd"] = t in beats
    return {"table": table, "pooled_sd": pooled_sd, "default_beta": DEFAULT_BETA, "default_score": default["score_mean"],
            "betas_beating_default": [t["beta"] for t in beats], "tied_after_beating": [t["beta"] for t in tied],
            "best_beta": best["beta"], "outcome": outcome, "complete": complete}


def main():
    runs = load_runs()
    assert not any(r["test_evaluated"] for r in runs), "beta screen runs must not evaluate the test set"
    dec = {t: decide(runs, t) for t in ["prostate", "breast"]}
    shared = None
    if all(d and d["best_beta"] is not None for d in dec.values()):
        ok = True
        for task, other in [("prostate", "breast"), ("breast", "prostate")]:
            tab = {t["beta"]: t for t in dec[task]["table"]}
            b, ob = dec[task]["best_beta"], dec[other]["best_beta"]
            ok &= ob in tab and abs(tab[b]["score_mean"] - tab[ob]["score_mean"]) <= dec[task]["pooled_sd"]
        shared = {"defensible": bool(ok), "rule": "each task's selected beta within one pooled s.d. of the other's (reported only)"}
    out = {"rule": {"score": "max centred 3-round mean validation DSC per run, mean over seeds",
                    "default_beta": DEFAULT_BETA,
                    "switch": "another beta must beat the default by more than the task's pooled seed s.d.",
                    "several": "highest score; ties within one pooled s.d. -> lower round-to-round q_i variance",
                    "revised": "2026-10-17 12:50, before any beta decision was computed"},
           "decision": dec, "shared_beta": shared, "runs": sorted(runs, key=lambda r: (r["task"], r["beta"], r["seed"]))}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "beta_decision.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    lines = ["| task | beta | seeds | val score mean | seed s.d. | margin over beta 0.5 | beats 0.5 by > pooled s.d. | q_i variance |",
             "|---|---|---|---|---|---|---|---|"]
    for task, d in dec.items():
        if not d:
            continue
        for t in d["table"]:
            star = " **(selected)**" if t["beta"] == d["best_beta"] else ""
            lines.append(f"| {task} | {t['beta']}{star} | {t['seeds']} | {t['score_mean']:.4f} | {t['score_sd']:.4f} | "
                         f"{t.get('margin_over_default', float('nan')):+.4f} | {t.get('beats_default_by_more_than_pooled_sd')} | "
                         f"{t['q_var_mean']:.2e} |")
        lines.append(f"| {task} | pooled s.d. = {d['pooled_sd']:.4f} | | | | | {d['outcome']} | |")
    with open(os.path.join(OUT, "beta_decision.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({t: (d and {k: d.get(k) for k in ["best_beta", "pooled_sd", "outcome", "complete"]})
                      for t, d in dec.items()}, indent=1), "\nshared:", shared)


if __name__ == "__main__":
    main()
