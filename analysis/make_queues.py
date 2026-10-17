"""
Write the experiment queues (plan.md section 0.0, order approved 17 Oct 2026) as queues/<name>.yaml.
beta comes from the task configs (set from results/beta/beta_decision.json before the headline).

    & $py analysis/make_queues.py            # all queues that can be generated now
Queues: headline, ablations, parity, fedprox_bounds, hetero, robustness_dp.
robustness_dp needs the headline's client update norms (DP clip = median update norm of the
headline XFedGA breast runs); it is skipped until those runs exist.

Robustness choices (flagged in plan.md): the noisy-q_i client and the corrupted-label client are
client index 1 (BUS-UCLM, the smaller breast client), where a false q_i = 1 or wrong labels can
move the aggregate most relative to its data size.
"""
import csv
import glob
import json
import os

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOTH = ["prostate", "breast"]
HEADLINE = ["fedavg", "fedga", "xfedga"]
ABLATIONS = ["abl_B", "abl_C", "abl_D", "abl_F", "random_search"]   # A = fedga, E = xfedga (headline)
SEEDS5, SEEDS3 = [1, 2, 3, 4, 5], [1, 2, 3]
ROBUST_CLIENT = 1
DP_NOISE_MULTIPLIER = 0.1


def entries(tasks, methods, seeds, sets=(), tag=None, **extra):
    out = []
    for seed in seeds:                      # seed-major order: every method/task gets seed 1 first
        for task in tasks:
            for m in methods:
                e = {"task": task, "method": m, "seed": seed}
                if sets:
                    e["set"] = list(sets)
                if tag:
                    e["tag"] = tag
                e.update(extra)
                out.append(e)
    return out


def headline_update_norm():
    norms = []
    for d in glob.glob(os.path.join(REPO, "results", "runs", "breast_xfedga_seed*")):
        if "_betascreen" in d or not os.path.exists(os.path.join(d, "run_meta.json")):
            continue
        with open(os.path.join(d, "round_log.csv"), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("update_norm_i"):
                    norms += json.loads(r["update_norm_i"])
    return float(np.median(norms)) if norms else None


def main():
    queues = {
        "headline": entries(BOTH, HEADLINE, SEEDS5),
        "ablations": entries(BOTH, ABLATIONS, SEEDS5),
        "parity": entries(["breast"], ["xfedga"], SEEDS3, tag="fl", runner="fl_dist"),
        "fedprox_bounds": entries(BOTH, ["fedprox"], SEEDS5) + entries(BOTH, ["centralized", "local_only"], SEEDS3),
        "hetero": entries(["prostate"], HEADLINE, SEEDS5, sets=["scenario.partition=acquisition"], tag="hetero"),
    }
    clip = headline_update_norm()
    if clip is not None:
        queues["robustness_dp"] = (
            entries(["breast"], ["xfedga"], SEEDS3, sets=[f"scenario.noisy_qi_client={ROBUST_CLIENT}"], tag="noisyq")
            + entries(["breast"], ["xfedga", "fedavg"], SEEDS3, sets=[f"scenario.corrupted_label_client={ROBUST_CLIENT}"],
                      tag="badlabels")
            + entries(["breast"], ["xfedga"], SEEDS3, sets=["scenario.dp.enabled=true", f"scenario.dp.clip_norm={clip:.4f}",
                                                            f"scenario.dp.noise_multiplier={DP_NOISE_MULTIPLIER}"], tag="dp"))
    os.makedirs(os.path.join(REPO, "queues"), exist_ok=True)
    for name, runs in queues.items():
        note = {"robustness_dp": f"DP clip_norm = median client update norm of the headline XFedGA breast runs ({clip})"}.get(name, "")
        with open(os.path.join(REPO, "queues", f"{name}.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({"name": name, "out": "results/runs", "note": note, "runs": runs}, f, sort_keys=False, width=200)
        print(f"{name}: {len(runs)} runs")
    if clip is None:
        print("robustness_dp: skipped (needs headline XFedGA breast update norms)")


if __name__ == "__main__":
    main()
