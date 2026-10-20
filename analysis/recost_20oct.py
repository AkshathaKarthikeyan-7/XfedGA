"""
Recost of the post-headline queue (items a-f, user order 19 Oct 2026) against the 27 Oct GPU deadline, overnight mode.

Per-round components (local_s, qi_s, ga_s, val_s, round_s; setup) are MEASURED from completed runs:
  breast v1:       FedAvg / FedGA / XFedGA = the v1 headline; XFedGA+ = the *_xplusscreen runs;
  prostate_ms v2:  FedGA / XFedGA / XFedGA+ = the *_msv2screen runs; FedAvg = *_final runs if finished.
Test time per run: breast from the v1 headline run_meta; prostate_ms from *_final run_meta if any, else breast x
  173 / 1127 test images (derived). Derived methods (labelled): FedAvg (if not measured) = FedGA - GA;
  ablation B = FedAvg + q_i; C = XFedGA - q_i; D = FedGA; F = XFedGA; random search = XFedGA - q_i; FedProx = FedAvg;
  centralized = FedAvg; local-only = FedAvg + (K - 1) validation passes; FedGA paper GA = FedGA + GA x (110/21 - 1),
  20 rounds; fl_dist = custom loop x 1.15.
Capacity (overnight mode, confirmed 18 Oct): full days DAY_HOURS_17OCT, other weekdays 16 h, from the projected end of
  the final headline + the XAI-metrics pass to 27 Oct 24:00; each day contributes in proportion to its remaining hours.
Contingency x1.2. Cut order (pre-registered, plan.md prostate_ms section): 1 FedProx/bounds -> 2 parity seeds 2-3 ->
  3 ablations to 2 seeds -> 4 H-ROB2 -> 5 FedGA paper-settings -> 6 (DSC-only control: n/a, the final headline is
  already running) -> 7 ablations dropped -> 8 dose-response p = 0.25. Never cut: dose-response p = 0.5 / 1.0, Flower demo.
Writes results/timing/recost_20oct.{json,md}.
    & $py analysis/recost_20oct.py
"""
import csv
import datetime as dt
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_costs import DAY_HOURS_17OCT  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "timing")
CONT = 1.2
OTHER_DAY_H = 16
DEADLINE = dt.datetime(2026, 10, 28, 0, 0)
K = {"breast": 2, "prostate_ms": 5}
SOURCES = {("breast", "fedavg"): "", ("breast", "fedga"): "", ("breast", "xfedga"): "",
           ("breast", "xfedga_plus"): "_xplusscreen", ("prostate_ms", "fedga"): "_msv2screen",
           ("prostate_ms", "xfedga"): "_msv2screen", ("prostate_ms", "xfedga_plus"): "_msv2screen",
           ("prostate_ms", "fedavg"): "_final"}


def measured(task, method, tag):
    rs = []
    for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed*")):
        if not re.fullmatch(rf"{task}_{method}_seed\d+_[0-9a-f]{{8}}{tag}", os.path.basename(d)):
            continue
        if not os.path.exists(os.path.join(d, "run_meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        c = {k: float(np.mean([float(x[k]) for x in rows])) for k in ["local_s", "qi_s", "ga_s", "val_s", "round_s"]}
        c["setup_s"] = meta["setup_s"]
        c["test_s"] = (meta["total_s"] - meta["setup_s"] - sum(float(x["round_s"]) for x in rows)) \
            if meta.get("test_evaluated") else None
        rs.append(c)
    if not rs:
        return None
    out = {k: float(np.mean([r[k] for r in rs])) for k in ["local_s", "qi_s", "ga_s", "val_s", "round_s", "setup_s"]}
    t = [r["test_s"] for r in rs if r["test_s"] is not None]
    out["test_s"] = float(np.mean(t)) if t else None
    out["n_runs"] = len(rs)
    return out


def costs():
    m = {k: measured(k[0], k[1], tag) for k, tag in SOURCES.items()}
    test = {"breast": m[("breast", "xfedga")]["test_s"]}
    pt = [v["test_s"] for (t, _), v in m.items() if t == "prostate_ms" and v and v["test_s"]]
    test["prostate_ms"] = float(np.mean(pt)) if pt else test["breast"] * 173 / 1127
    derived = {"prostate_ms_test": not pt}
    r = {}
    for t in K:
        fg, xg, xp = m[(t, "fedga")], m[(t, "xfedga")], m[(t, "xfedga_plus")]
        fa = m.get((t, "fedavg"))
        fedavg = fa["round_s"] if fa else fg["round_s"] - fg["ga_s"]
        derived[f"{t}_fedavg"] = fa is None
        qi = xg["qi_s"]
        r[t] = {"fedavg": fedavg, "fedga": fg["round_s"], "xfedga": xg["round_s"], "xfedga_plus": xp["round_s"],
                "abl_B": fedavg + qi, "abl_C": xg["round_s"] - qi, "abl_D": fg["round_s"], "abl_F": xg["round_s"],
                "random_search": xg["round_s"] - qi, "fedprox": fedavg, "centralized": fedavg,
                "local_only": fedavg + (K[t] - 1) * fg["val_s"],
                "fedga_paper": fg["round_s"] + fg["ga_s"] * (110 / 21 - 1), "xfedga_plus_dsc": xp["round_s"]}
        r[t]["_setup"] = fg["setup_s"]
    return r, test, derived, m


def run_h(r, test, task, method, seeds, rounds=15, factor=1.0):
    return factor * seeds * (r[task]["_setup"] + rounds * r[task][method] + test[task]) / 3600


def items(r, test):
    T = ["breast", "prostate_ms"]
    dose = ["fedavg", "fedga", "xfedga", "xfedga_plus"]
    per_p = sum(run_h(r, test, "breast", m, 3) for m in dose)
    return [
        ("xai_final", "XAI metrics on the 35 final checkpoints (no training; ~3 min per run measured 18 Oct)", 35 * 3 / 60),
        ("dose_p050_p100", "a. dose-response breast p = 0.5, 1.0 x 4 methods x 3 seeds", 2 * per_p),
        ("dose_p025", "a. dose-response breast p = 0.25 x 4 methods x 3 seeds", per_p),
        ("flower_demo", "b. fl_dist breast XFedGA+ and XFedGA, seed 1 (demo)",
         sum(run_h(r, test, "breast", m, 1, factor=1.15) for m in ["xfedga", "xfedga_plus"])),
        ("parity", "b. fl_dist breast XFedGA+ and XFedGA, seeds 2-3 (parity)",
         sum(run_h(r, test, "breast", m, 2, factor=1.15) for m in ["xfedga", "xfedga_plus"])),
        ("fedga_paper", "c. FedGA paper GA settings (P 10, G 20, 20 rounds), 2 tasks x 3 seeds",
         sum(run_h(r, test, t, "fedga_paper", 3, rounds=20) for t in T)),
        ("h_rob2", "d. H-ROB2 noisy-q_i client, breast, XFedGA and XFedGA+ x 3 seeds",
         sum(run_h(r, test, "breast", m, 3) for m in ["xfedga", "xfedga_plus"])),
        ("ablations", "e. ablations B, C, D, F + random search, 2 tasks x 3 seeds",
         sum(run_h(r, test, t, m, 3) for t in T for m in ["abl_B", "abl_C", "abl_D", "abl_F", "random_search"])),
        ("fedprox_bounds", "f. FedProx 2 tasks x 5 seeds; centralized + local-only 2 tasks x 3 seeds",
         sum(run_h(r, test, t, "fedprox", 5) + run_h(r, test, t, "centralized", 3) + run_h(r, test, t, "local_only", 3)
             for t in T)),
    ]


CUTS = [("fedprox_bounds", 0.0), ("parity", 0.0), ("ablations", 2 / 3), ("h_rob2", 0.0), ("fedga_paper", 0.0),
        ("ablations", 0.0), ("dose_p025", 0.0)]


def headline_eta(r, test):
    st = json.load(open(os.path.join(RUNS, "_queues", "final_headline_status.json"), encoding="utf-8"))
    q = __import__("yaml").safe_load(open(os.path.join(REPO, "queues", "final_headline.yaml"), encoding="utf-8"))
    done = len(st["runs"])
    left = q["runs"][done:]
    hours = sum(run_h(r, test, e["task"], e["method"], 1) for e in left)
    return dt.datetime.now() + dt.timedelta(hours=hours), done, len(q["runs"]), hours


def capacity(start):
    cap, t = 0.0, start
    while t < DEADLINE:
        day_end = dt.datetime.combine(t.date() + dt.timedelta(days=1), dt.time())
        seg = (min(day_end, DEADLINE) - t).total_seconds() / 3600
        cap += seg / 24 * DAY_HOURS_17OCT.get(t.date(), OTHER_DAY_H)
        t = day_end
    return round(cap, 1)


def main():
    r, test, derived, m = costs()
    eta, done, total, left_h = headline_eta(r, test)
    it = items(r, test)
    start = eta + dt.timedelta(hours=it[0][2])
    cap = capacity(start)
    orig = {k: v * CONT for k, _, v in it}
    h = dict(orig)
    tot = sum(h.values()) - h["xai_final"]
    applied = []
    for k, keep in CUTS:
        if tot <= cap:
            break
        new = orig[k] * keep
        if h[k] == 0 or new >= h[k]:
            continue
        applied.append({"item": k, "to": "dropped" if keep == 0 else f"x{keep:.2f}", "saved_h": h[k] - new})
        tot -= h[k] - new
        h[k] = new
    out = {"final_headline": {"runs_done": done, "runs_total": total, "hours_left_est": left_h, "eta": str(eta)},
           "capacity_from": str(start), "capacity_h": cap, "test_s": test, "derived": derived,
           "round_s": r, "items": [{"item": k, "label": lab, "h_with_contingency": orig[k], "after_cuts_h": h[k]}
                                   for k, lab, _ in it],
           "total_after_cuts_h": tot, "cuts_applied": applied, "fits": tot <= cap}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "recost_20oct.json"), "w", encoding="utf-8"), indent=1, default=str)
    L = [f"Final headline: {done}/{total} runs done; about {left_h:.1f} h left; ETA {eta:%a %d %b %H:%M}. "
         f"Capacity from {start:%a %d %b %H:%M} to 27 Oct 24:00 (overnight mode): {cap} GPU-h.", "",
         "| Item | Hours (×1.2) | After cuts |", "|---|---|---|"]
    for k, lab, _ in it:
        L.append(f"| {lab} | {orig[k]:.1f} | {h[k]:.1f} |")
    L += ["", f"Total a–f after cuts: {tot:.1f} h of {cap} h → {'fits' if tot <= cap else 'DOES NOT FIT'}. Cuts applied: "
          + (", ".join(f"{a['item']} {a['to']}" for a in applied) or "none") + ".",
          "Derived (not measured): " + ", ".join(k for k, v in derived.items() if v) + "; ablations, FedProx, bounds, "
          "paper-GA and fl_dist costs are derived from measured components (see the script docstring)."]
    open(os.path.join(OUT, "recost_20oct.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
