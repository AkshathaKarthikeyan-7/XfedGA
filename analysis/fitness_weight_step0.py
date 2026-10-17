"""
Fitness-weight Step 0 (spec section 5B; decision 17 Oct 2026: weights stay (0.5, 0.3, 0.2), Step 0
reports sensitivity). Offline, no GPU: re-ranks every logged GA candidate pool
(<run>/ga_candidates.jsonl, one line per round with each evaluated candidate's DSC, HD95_norm, EC)
under every point of the weight simplex (step 0.1, w >= 0, sum 1) and reports how often the winner
differs from the winner under the default weights.

The default winner is the argmax over all evaluated candidates; with elitism the best-ever candidate
is in the final population, so this is exactly the candidate the GA returned (checked against the
logged per-generation best). The weighted sum is a scalarisation choice; this measures how much the
choice matters for selection within a round, not for the GA trajectory.

    & $py analysis/fitness_weight_step0.py --runs "results/runs/*_xfedga_seed*"
Outputs: results/fitness_weights/step0.json (per weight point: share of rounds whose winner changes,
per task) and step0_grid.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "fitness_weights")
DEFAULT = (0.5, 0.3, 0.2)


def simplex(step=0.1):
    n = int(round(1 / step))
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def winner(cands, w):
    scores = [w[0] * c["dsc"] - w[1] * c["hd95_norm"] + w[2] * c["ec"] for c in cands]
    return int(np.argmax(scores))


def load_pools(pattern):
    pools = []
    for d in sorted(glob.glob(pattern)):
        path = os.path.join(d, "ga_candidates.jsonl")
        meta_path = os.path.join(d, "run_meta.json")
        if not (os.path.exists(path) and os.path.exists(meta_path)) or not uses_current_cam(d):
            continue
        meta = json.load(open(meta_path, encoding="utf-8"))
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["candidates"] and "ec" in rec["candidates"][0]:
                    pools.append({"run_id": meta["run_id"], "task": meta["task"], "round": rec["round"],
                                  "cands": rec["candidates"]})
    return pools


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(REPO, "results", "runs", "*_xfedga_seed*"))
    args = ap.parse_args()
    pools = load_pools(args.runs)
    if not pools:
        print("no candidate pools found")
        return
    grid = simplex()
    rows = []
    for w in grid:
        row = {"w1": w[0], "w2": w[1], "w3": w[2]}
        for task in sorted({p["task"] for p in pools}):
            ps = [p for p in pools if p["task"] == task]
            changed = [winner(p["cands"], w) != winner(p["cands"], DEFAULT) for p in ps]
            row[f"{task}_share_winner_changed"] = float(np.mean(changed))
            row[f"{task}_rounds"] = len(ps)
        rows.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "step0_grid.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    tasks = sorted({p["task"] for p in pools})
    summary = {"default_weights": DEFAULT, "grid_points": len(grid), "runs": sorted({p["run_id"] for p in pools}),
               "per_task": {t: {"rounds": sum(p["task"] == t for p in pools),
                                "mean_share_changed_over_grid": float(np.mean([r[f"{t}_share_winner_changed"] for r in rows])),
                                "corners": {f"{c}": next(r[f"{t}_share_winner_changed"] for r in rows
                                                         if (r["w1"], r["w2"], r["w3"]) == c)
                                            for c in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]}}
                            for t in tasks}}
    with open(os.path.join(OUT, "step0.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary["per_task"], indent=1))


if __name__ == "__main__":
    main()
