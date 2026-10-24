"""
Recost on Sat 24 Oct 2026 (resume after the user pause): the remaining runs of item e (ablations) and item f (FedProx
+ bounds), one queue, against 27 Oct 24:00, overnight mode. Remaining runs = queue entries without run_meta.json.
Costs: analysis/recost_20oct.py (measured components; ablation / FedProx / bounds costs derived there).
Pre-registered cut order for e / f if needed: f dropped -> e to 2 seeds -> e dropped (reported as a limitation).
Writes results/timing/recost_24oct.{json,md}.
    & $py analysis/recost_24oct.py
"""
import datetime as dt
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recost_20oct import CONT, capacity, costs, run_h  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def remaining(queue):
    out = []
    for e in yaml.safe_load(open(os.path.join(REPO, "queues", f"post_{queue}.yaml"), encoding="utf-8"))["runs"]:
        pat = os.path.join(REPO, "results", "runs", f"{e['task']}_{e['method']}_seed{e['seed']}_*_{e['tag']}")
        if not any(os.path.exists(os.path.join(d, "run_meta.json")) for d in glob.glob(pat)):
            out.append(e)
    return out


def main():
    r, test, _, _ = costs()
    left = {q: remaining(q) for q in ["e_ablations", "f_fedprox_bounds"]}
    hrs = {q: sum(run_h(r, test, e["task"], e["method"], 1) for e in v) for q, v in left.items()}
    cap = capacity(dt.datetime.now())
    tot = sum(hrs.values()) * CONT
    now, eta = dt.datetime.now(), {}
    acc = now
    for q in hrs:
        acc += dt.timedelta(hours=hrs[q] * CONT)
        eta[q] = acc
    out = {"computed": str(now), "capacity_h": cap, "remaining_runs": {q: len(v) for q, v in left.items()},
           "hours_with_contingency": {q: hrs[q] * CONT for q in hrs}, "total_h": tot, "fits": tot <= cap,
           "eta": {q: str(t) for q, t in eta.items()}}
    json.dump(out, open(os.path.join(REPO, "results", "timing", "recost_24oct.json"), "w", encoding="utf-8"), indent=1)
    L = [f"Capacity {now:%a %d %b %H:%M} → 27 Oct 24:00 (overnight mode): {cap} GPU-h.", "",
         "| Item | Remaining runs | Hours (×1.2) | ETA |", "|---|---|---|---|"] + \
        [f"| {q} | {len(left[q])} | {hrs[q] * CONT:.1f} | {eta[q]:%a %d %b %H:%M} |" for q in hrs] + \
        ["", f"Total {tot:.1f} h of {cap} h → {'fits, no cuts' if tot <= cap else 'DOES NOT FIT'}."]
    open(os.path.join(REPO, "results", "timing", "recost_24oct.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
