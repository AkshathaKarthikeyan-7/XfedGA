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
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_filter import uses_current_cam  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "fitness_weights")
DEFAULT = (0.5, 0.3, 0.2)
NAMED = [(0.5, 0.3, 0.2), (0.7, 0.2, 0.1), (0.8, 0.2, 0.0), (1.0, 0.0, 0.0)]   # 18 Oct 2026 request (tuning stage)
HEADLINE = os.path.join(REPO, "results", "runs", "*_xfedga_seed[1-5]_[0-9a-f]*")


def simplex(step=0.1):
    n = int(round(1 / step))
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


def winner(cands, w):
    scores = [w[0] * c["dsc"] - w[1] * c["hd95_norm"] + w[2] * c["ec"] for c in cands]
    return int(np.argmax(scores))


def load_pools(pattern, include_tagged=False):
    pools = []
    for d in sorted(glob.glob(pattern)):
        if not include_tagged and not re.fullmatch(r".+_seed\d+_[0-9a-f]{8}", os.path.basename(d)):
            continue                                   # tagged runs (betascreen, v2screen, fl, ...) excluded
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
    ap.add_argument("--runs", default=HEADLINE, help="default: the v1 headline XFedGA runs (untagged)")
    ap.add_argument("--include_tagged", action="store_true", help="also use tagged runs matching --runs")
    args = ap.parse_args()
    pools = load_pools(args.runs, args.include_tagged)
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
    named, lines = {}, ["| Task | Weights (w1, w2, w3) | Rounds | Winner changes vs (0.5, 0.3, 0.2) | "
                        "Winner DSC (fitness subset) | Winner HD95_norm | Winner EC | Winner = w_agg |",
                        "|---|---|---|---|---|---|---|---|"]
    for t in tasks:
        ps = [p for p in pools if p["task"] == t]
        for w in NAMED:
            win = [p["cands"][winner(p["cands"], w)] for p in ps]
            ch = [winner(p["cands"], w) != winner(p["cands"], DEFAULT) for p in ps]
            kept = [winner(p["cands"], w) == 0 for p in ps]          # candidate 0 = w_agg (first evaluated)
            named[f"{t}/{w}"] = {"rounds": len(ps), "share_winner_changed": float(np.mean(ch)),
                                 "n_changed": int(np.sum(ch)), "share_w_agg_kept": float(np.mean(kept)),
                                 **{f"winner_mean_{k}": float(np.mean([c[k] for c in win])) for k in ("dsc", "hd95_norm", "ec")}}
            n = named[f"{t}/{w}"]
            lines.append(f"| {t} | {w} | {len(ps)} | {n['n_changed']} ({n['share_winner_changed']:.0%}) | "
                         f"{n['winner_mean_dsc']:.4f} | {n['winner_mean_hd95_norm']:.4f} | {n['winner_mean_ec']:.4f} | "
                         f"{n['share_w_agg_kept']:.0%} |")
    lines += ["", "Re-ranking of the logged candidate pools of the v1 headline XFedGA runs (CPU, no new evaluation). "
              "Within-round selection only: the GA trajectory under other weights is not simulated. Winner metrics are on "
              "the 40-image fitness subset, not validation."]
    summary["named"] = named
    with open(os.path.join(OUT, "step0.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(OUT, "step0_named.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(summary["per_task"], indent=1))


if __name__ == "__main__":
    main()
