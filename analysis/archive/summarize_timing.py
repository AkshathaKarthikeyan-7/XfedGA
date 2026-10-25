"""
Summarize timing_events.jsonl from every run folder under results/timing/runs/
into results/timing/timing_summary.csv and timing_summary.json.

Per run: rounds completed, setup time, mean/min/max wall clock per round,
mean local-training / q_i / GA / fitness-eval / OOD-eval seconds per round,
fitness evaluations per round, seconds per fitness evaluation, peak GPU
memory (PyTorch allocated / reserved), and the 15-round projection
(setup + 15 x mean round). If the run stopped early, "extrapolated" is True.

Usage (from repo root):
    python analysis/summarize_timing.py
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "timing")
KEYS = ["wall_s", "local_train_s", "qi_s", "agg_s", "ga_s", "fitness_eval_s", "ga_overhead_s", "ood_eval_s"]


def summarize(path):
    events = [json.loads(line) for line in open(path)]
    start = next(e for e in events if e["event"] == "start")
    setup = next((e["setup_s"] for e in events if e["event"] == "setup_done"), float("nan"))
    rounds = [e for e in events if e["event"] == "round"]
    end = next((e for e in events if e["event"] == "end"), None)
    if not rounds:
        return None
    rec = {"run": os.path.basename(os.path.dirname(path)), "task": start["task"], "mode": start["mode"],
           "argv": " ".join(start["argv"]), "started": start["time"],
           "status": end["status"] if end else "running_or_crashed",
           "rounds_completed": len(rounds), "setup_s": setup}
    for k in KEYS:
        v = np.array([r[k] for r in rounds])
        rec[f"mean_{k}"] = float(v.mean())
    w = np.array([r["wall_s"] for r in rounds])
    rec["min_wall_s"], rec["max_wall_s"] = float(w.min()), float(w.max())
    rec["fitness_evals_per_round"] = float(np.mean([r["n_fitness_evals"] for r in rounds]))
    rec["s_per_fitness_eval"] = rec["mean_fitness_eval_s"] / max(rec["fitness_evals_per_round"], 1)
    rec["ga_share_of_round"] = rec["mean_ga_s"] / rec["mean_wall_s"]
    rec["peak_alloc_MiB"] = float(max(r.get("peak_alloc_MiB", 0) for r in rounds))
    rec["peak_reserved_MiB"] = float(max(r.get("peak_reserved_MiB", 0) for r in rounds))
    rec["projected_15_round_h"] = (setup + 15 * rec["mean_wall_s"]) / 3600
    rec["extrapolated"] = len(rounds) < 15
    return rec


def main():
    recs = [r for r in (summarize(p) for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "timing_events.jsonl")))) if r]
    with open(os.path.join(HERE, "timing_summary.json"), "w") as f:
        json.dump(recs, f, indent=2)
    with open(os.path.join(HERE, "timing_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    for r in recs:
        print(f"{r['run']:28s} rounds={r['rounds_completed']:2d} wall/round={r['mean_wall_s']:7.1f}s "
              f"local={r['mean_local_train_s']:6.1f}s qi={r['mean_qi_s']:5.1f}s GA={r['mean_ga_s']:6.1f}s "
              f"evals={r['fitness_evals_per_round']:.0f} s/eval={r['s_per_fitness_eval']:5.2f} "
              f"ood={r['mean_ood_eval_s']:5.1f}s peak={r['peak_reserved_MiB']:.0f}MiB "
              f"proj15={r['projected_15_round_h']:.2f}h extrapolated={r['extrapolated']} [{r['status']}]")


if __name__ == "__main__":
    main()
