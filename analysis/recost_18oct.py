"""
Recost of the master plan of 18 Oct 2026 (with decision B: prostate_ms is the main prostate experiment) at MEASURED
speeds, against the 27 Oct GPU deadline.

Per-run cost = setup + rounds x per-round time + test/selection time.
Measured (round_log.csv per-round components local_s / qi_s / ga_s / val_s / round_s; run_meta.json setup / total):
  breast   v1: headline runs; v2: *_v2screen runs (FedGA, XFedGA);
  prostate v1 / v2 (PROMISE12 -> Decathlon; its RESULTS are superseded, its TIMINGS are valid compute measurements).
Derived (labelled "derived"):
  prostate_ms (5 clients, BMC server) from the prostate components: local x client slices (1483 / 778),
    q_i x 5/2 clients, GA x fitness-set size (v1: 40 / 40; v2: 106 / 117), validation x 105 / 117, setup x 1.5,
    test x 173 / 241 slices;
  FedAvg v2 = FedGA v2 round - GA time; XFedGA+ = XFedGA + K client-score evaluations (K x GA time / evaluations)
    + K client validation passes (RC4 logging; one pass = validation time / 2); FedGA paper GA = FedGA + GA time x
    (110 / 21 - 1), 20 rounds; test time from the v1 headline.
Capacity, two modes; the user confirmed 'overnight' on 18 Oct 2026 (laptop on overnight and during weekdays):
  evenings  : DAY_HOURS_17OCT full days; other days EVENING_HOURS (assumed); tonight TONIGHT_H (assumed);
  overnight : full days unchanged; other weekdays 20 h minus a 4 h buffer = 16 h; tonight = from the end of the v2
              screen (queue log) to midnight.
To 27 Oct inclusive. Contingency x1.2.
Cut order (applied in order, only while the plan does not fit): 1 FedProx/bounds dropped -> 2 parity seeds 2-3 dropped
-> 3 ablations to 2 seeds -> 4 H-ROB2 dropped -> 5 FedGA paper-settings check dropped -> 6 XFedGA+ (DSC-only) control
3 seeds in the final headline (it is never in the dose-response) -> 7 ablations dropped -> 8 dose-response p = 0.25
dropped. Never cut: final headline primary arms (5 seeds), prostate_ms, dose-response p = 0.5 and 1.0, Flower demo.
Writes results/timing/recost_18oct.json / .md.
    & $py analysis/recost_18oct.py
"""
import csv
import datetime as dt
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_costs import DAY_HOURS_17OCT, EVENING_HOURS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "timing")
CONTINGENCY = 1.2
TONIGHT_H = 4
WEEKDAY_OVERNIGHT_H = 20 - 4
MODES = ["overnight"]                 # confirmed by the user, 18 Oct 2026
DEADLINE = dt.date(2026, 10, 27)
TASKS = ["breast", "prostate_ms"]
MS_SCALE = {"local": 1483 / 778, "qi": 5 / 2, "ga": {"v1": 1.0, "v2": 106 / 117}, "val": 105 / 117,
            "setup": 1.5, "test": 173 / 241, "clients": 5}
# v2-screen runs whose wall-clock overlapped other work on this machine (18 Oct 2026): times NOT clean.
# A clean run of the same task/method is preferred; if none exists the contaminated mean is used and labelled.
CONTAMINATED = {
    "prostate_xfedga_seed1_2a059f3e_v2screen": "GPU shared with a full pytest run (15:50-16:05) and slow tests (16:05-16:09)",
    "breast_fedga_seed1_019627d6_v2screen": "18 s GPU known-answer test; CPU CAM check",
    "breast_xfedga_seed1_bb69b34e_v2screen": "CPU-heavy multi-site leakage audit and breast duplicate audit",
    "prostate_fedga_seed2_5666ea23_v2screen": "CPU: prostate_ms cache build and CPU test runs",
    "prostate_xfedga_seed2_2a059f3e_v2screen": "CPU: EC tests, FOV-crop derivation, prostate_ms v1/v2 cache builds",
    "breast_xfedga_seed2_bb69b34e_v2screen": "GPU throttled: median SM clock 1335 MHz (queue log)",
}
COMP = ["local_s", "qi_s", "ga_s", "val_s", "round_s"]


def measured(task, method, tag):
    rs, dirty = [], []
    for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed*")):
        if not re.fullmatch(rf"{task}_{method}_seed\d+_[0-9a-f]{{8}}{tag}", os.path.basename(d)):
            continue
        if not os.path.exists(os.path.join(d, "run_meta.json")):
            continue
        meta = json.load(open(os.path.join(d, "run_meta.json"), encoding="utf-8"))
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        col = lambda k: float(np.mean([float(x[k]) for x in rows]))
        rec = {**{k: col(k) for k in COMP}, "evals": col("fitness_evals"), "setup_s": meta["setup_s"],
               "test_s": meta["total_s"] - meta["setup_s"] - sum(float(x["round_s"]) for x in rows)}
        (dirty if os.path.basename(d) in CONTAMINATED else rs).append(rec)
    use, clean = (rs, True) if rs else (dirty, False)
    if not use:
        return None
    return {k: float(np.mean([r[k] for r in use])) for k in use[0]} | {"n_runs": len(use), "clean": clean}


def method_rounds(src, v1, proto, k_clients, scale=None):
    """Per-round seconds of every method from measured FedGA / XFedGA components (optionally scaled)."""
    sc = scale or {"local": 1, "qi": 1, "ga": {"v1": 1, "v2": 1}, "val": 1}
    fg, xg = src["fedga"], src["xfedga"]

    def rnd(m, with_qi, with_ga):
        over = m["round_s"] - sum(m[k] for k in ["local_s", "qi_s", "ga_s", "val_s"])
        return (m["local_s"] * sc["local"] + (m["qi_s"] * sc["qi"] if with_qi else 0)
                + (m["ga_s"] * sc["ga"][proto] if with_ga else 0) + m["val_s"] * sc["val"] + over)
    r = {"fedga": rnd(fg, False, True), "xfedga": rnd(xg, True, True), "fedavg": rnd(fg, False, False)}
    if proto == "v1" and scale is None and v1.get("fedavg"):
        r["fedavg"] = v1["fedavg"]["round_s"]                      # measured
    ga_eval = xg["ga_s"] * sc["ga"][proto] / xg["evals"]
    val_pass = xg["val_s"] * sc["val"] / 2
    r["xfedga_plus"] = r["xfedga"] + k_clients * ga_eval + k_clients * val_pass
    r["fedga_paper"] = r["fedga"] + fg["ga_s"] * sc["ga"][proto] * (110 / 21 - 1)
    # acceptance check (pre-registered 18 Oct 2026): GA on half A (about half the GA time) + 2 evaluations on half B
    ga_x = xg["ga_s"] * sc["ga"][proto]
    r["xfedga_acc"] = r["xfedga"] - 0.5 * ga_x + 2 * 0.5 * ga_eval
    r["xfedga_plus_acc"] = r["xfedga_plus"] - 0.5 * ga_x + 2 * 0.5 * ga_eval
    return r


def costs():
    c = {}
    for t in ["breast", "prostate"]:
        v1 = {m: measured(t, m, "") for m in ["fedavg", "fedga", "xfedga"]}
        v2 = {m: measured(t, m, "_v2screen") for m in ["fedga", "xfedga"]}
        test = {m: v1[m]["test_s"] for m in v1}
        for proto, src in [("v1", v1), ("v2", v2)]:
            if any(src.get(m) is None for m in ["fedga", "xfedga"]):
                continue
            clean = {m: src[m]["clean"] for m in ["fedga", "xfedga"]}
            if t == "breast":
                c[("breast", proto)] = {"round_s": method_rounds(src, v1, proto, 2), "setup_s": src["fedga"]["setup_s"],
                                        "test_s": test, "clean": clean,
                                        "derived": ["xfedga_plus", "fedga_paper"] + (["fedavg"] if proto == "v2" else [])}
            else:
                c[("prostate_ms", proto)] = {
                    "round_s": method_rounds(src, v1, proto, MS_SCALE["clients"], MS_SCALE),
                    "setup_s": src["fedga"]["setup_s"] * MS_SCALE["setup"],
                    "test_s": {m: v * MS_SCALE["test"] for m, v in test.items()}, "clean": clean,
                    "derived": ["fedavg", "fedga", "xfedga", "xfedga_plus", "fedga_paper"]}
    return c


def run_h(c, task, proto, method, seeds, rounds=15, test=True, factor=1.0):
    k = c[(task, proto)]
    base = {"xfedga_plus": "xfedga", "fedga_paper": "fedga", "xfedga_acc": "xfedga",
            "xfedga_plus_acc": "xfedga"}.get(method, method)
    t = k["test_s"].get(base, k["test_s"]["xfedga"]) if test else 0.0
    return factor * seeds * (k["setup_s"] + rounds * k["round_s"][method] + t) / 3600


def plan(c, proto, xplus, accept=False):
    """proto: task -> 'v1' | 'v2' (adopted). Items in queue order."""
    P = lambda t: proto[t]
    items = [
        ("screens", "prostate_ms protocol + XFedGA+ screen (FedGA, XFedGA, XFedGA+ x2 variants; v1 + v2; 2 seeds; no test)"
                    " + breast XFedGA+ screen (2 variants x 2 seeds, adopted breast protocol)",
         sum(run_h(c, "prostate_ms", p, m, 2, test=False) for p in ("v1", "v2")
             for m in ["fedga", "xfedga", "xfedga_plus", "xfedga_plus"])
         + run_h(c, "breast", P("breast"), "xfedga_plus", 2 * 2, test=False)),
    ]
    head = sum(run_h(c, "prostate_ms", P("prostate_ms"), m, 5) for m in ["fedavg", "fedga", "xfedga"])
    if P("breast") == "v2":    # breast v1 kept -> the v1 breast headline is the final breast headline (no rerun)
        head += sum(run_h(c, "breast", "v2", m, 5) for m in ["fedavg", "fedga", "xfedga"])
    items.append(("final_headline", "primary arms: prostate_ms FedAvg / FedGA / XFedGA x 5 seeds"
                  + (" + breast (v2) x 5 seeds" if P("breast") == "v2" else " (breast v1 headline reused)"), head))
    items.append(("headline_xplus", "XFedGA+ x 5 seeds, both tasks (if adopted)",
                  sum(run_h(c, t, P(t), "xfedga_plus", 5) for t in TASKS) if xplus else 0.0))
    items.append(("headline_xplus_dsc", "XFedGA+ (DSC-only) control x 5 seeds, both tasks (if adopted)",
                  sum(run_h(c, t, P(t), "xfedga_plus", 5) for t in TASKS) if xplus else 0.0))
    acc_m = "xfedga_plus_acc" if xplus else "xfedga_acc"     # acceptance version of the method selected per task
    items.append(("acceptance_screen", "acceptance-check version of the selected method, 2 tasks x 2 seeds, no test",
                  sum(run_h(c, t, P(t), acc_m, 2, test=False) for t in TASKS)))
    items.append(("headline_accept", "acceptance-check arm x 5 seeds, both tasks (if adopted; not in the cut order)",
                  sum(run_h(c, t, P(t), acc_m, 5) for t in TASKS) if accept else 0.0))
    dm = ["fedavg", "fedga", "xfedga"] + (["xfedga_plus"] if xplus else [])     # the DSC-only control is not run here
    per_p = sum(run_h(c, "breast", P("breast"), m, 3) for m in dm)
    items.append(("dose_p050_p100", f"dose-response breast p = 0.5, 1.0 x {len(dm)} methods x 3 seeds (p = 0 reused)",
                  2 * per_p))
    items.append(("dose_p025", f"dose-response breast p = 0.25 x {len(dm)} methods x 3 seeds", per_p))
    items.append(("fedga_paper", "FedGA (paper GA settings: P 10, G 20, 20 rounds; batch 8 declared), 2 tasks x 3 seeds",
                  sum(run_h(c, t, P(t), "fedga_paper", 3, rounds=20) for t in TASKS)))
    items.append(("h_rob2", "noisy-q_i client: XFedGA, breast, 3 seeds", run_h(c, "breast", P("breast"), "xfedga", 3)))
    abl = sum(run_h(c, t, P(t), m, 3) for t in TASKS for m in ["xfedga", "fedga", "fedga", "xfedga", "xfedga"])
    items.append(("ablations", "ablations B, C, D, F + random search, 2 tasks x 3 seeds (A, E from the headline)", abl))
    items.append(("parity", "Flower fl_dist XFedGA, breast, 3 seeds (x1.15); seed 1 = Flower demo",
                  run_h(c, "breast", P("breast"), "xfedga", 3, factor=1.15)))
    items.append(("fedprox_bounds", "FedProx 2 tasks x 5 seeds; centralized + local-only 2 tasks x 3 seeds",
                  sum(run_h(c, t, P(t), "fedavg", 5) + run_h(c, t, P(t), "fedavg", 2 * 3) for t in TASKS)))
    cuts = [("fedprox_bounds", 0.0), ("parity", 1 / 3), ("ablations", 2 / 3), ("h_rob2", 0.0), ("fedga_paper", 0.0),
            ("headline_xplus_dsc", 3 / 5), ("ablations", 0.0), ("dose_p025", 0.0)]   # keep = share of the original
    return items, cuts


def screen_end():
    log = os.path.join(RUNS, "_queues", "v2_screen.log")
    ends = [ln[:19] for ln in open(log, encoding="utf-8") if "queue v2_screen finished" in ln] if os.path.exists(log) else []
    return dt.datetime.strptime(ends[-1], "%Y-%m-%d %H:%M:%S") if len(ends) >= 2 else dt.datetime.now()
    # the first "finished" line in the log is the 14:10 dry run


def capacity(mode, start=dt.date(2026, 10, 19)):
    if mode == "evenings":
        tonight, other = TONIGHT_H, EVENING_HOURS
    else:
        end = screen_end()
        midnight = dt.datetime.combine(end.date() + dt.timedelta(days=1), dt.time())
        tonight, other = max(0.0, (midnight - end).total_seconds() / 3600), WEEKDAY_OVERNIGHT_H
    d, cap = start, tonight
    while d <= DEADLINE:
        cap += DAY_HOURS_17OCT.get(d, other)
        d += dt.timedelta(days=1)
    return round(cap, 1)


def evaluate(c, proto, xplus, mode, accept=False):
    items, cuts = plan(c, proto, xplus, accept)
    cap = capacity(mode)
    orig = {k: v * CONTINGENCY for k, _, v in items}
    h = dict(orig)
    total = sum(h.values())
    applied = []
    for k, keep in cuts:                       # pre-agreed order, only while it does not fit
        if total <= cap:
            break
        new = orig[k] * keep
        if h[k] == 0 or new >= h[k]:
            continue
        applied.append({"item": k, "saved_h": h[k] - new, "to": "dropped" if keep == 0 else f"x{keep:.2f}"})
        total -= h[k] - new
        h[k] = new
    return {"protocol": proto, "xfedga_plus_adopted": xplus, "acceptance_adopted": accept, "capacity_mode": mode, "capacity_h": cap,
            "items": [{"item": k, "label": lab, "raw_h": v, "with_contingency_h": v * CONTINGENCY} for k, lab, v in items],
            "total_before_cuts_h": sum(v * CONTINGENCY for _, _, v in items), "cuts_applied": applied,
            "total_after_cuts_h": total, "fits": total <= cap}


def main():
    c = costs()
    need = [(t, p) for t in TASKS for p in ("v1", "v2")]
    if not all(k in c for k in need):
        print("timings incomplete:", sorted(c))
        return
    dec_path = os.path.join(REPO, "results", "protocol_v2", "v2_adoption.json")
    breast_opts = ["v1", "v2"]
    if os.path.exists(dec_path):
        d = json.load(open(dec_path, encoding="utf-8"))["decision"].get("breast")
        if d and d.get("complete"):
            breast_opts = ["v2" if d["adopt_v2"] else "v1"]
    out = {"round_costs_s": {f"{t}/{p}": v for (t, p), v in c.items()}, "assumptions": {
        "tonight_h": TONIGHT_H, "evening_h": EVENING_HOURS, "full_days": {str(k): v for k, v in DAY_HOURS_17OCT.items()},
        "deadline": str(DEADLINE), "contingency": CONTINGENCY, "prostate_ms_scaling": MS_SCALE},
        "contaminated_runs": CONTAMINATED, "scenarios": {}}
    for mode, b, ms, xp, ac in itertools.product(MODES, breast_opts, ["v1", "v2"], [False, True], [False, True]):
        name = (f"{mode}: breast {b}, prostate_ms {ms}, XFedGA+ " + ("adopted" if xp else "not adopted")
                + ", acceptance " + ("adopted" if ac else "not adopted"))
        out["scenarios"][name] = evaluate(c, {"breast": b, "prostate_ms": ms}, xp, mode, ac)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "recost_18oct.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    L = ["| Task / protocol | FedAvg | FedGA | XFedGA | XFedGA+ | FedGA paper GA | XFedGA acc. | XFedGA+ acc. | Clean measurement? |",
         "|---|---|---|---|---|---|---|---|---|"]
    for (t, p), v in sorted(c.items()):
        r, dv = v["round_s"], v["derived"]
        lab = lambda m: f"{r[m]:.0f} s" + (" (derived)" if m in dv else "")
        L.append(f"| {t} {p} | {lab('fedavg')} | {lab('fedga')} | {lab('xfedga')} | {lab('xfedga_plus')} | "
                 f"{lab('fedga_paper')} | {r['xfedga_acc']:.0f} s (derived) | {r['xfedga_plus_acc']:.0f} s (derived) | "
                 + ", ".join(f"{m}: {'yes' if ok else 'NO'}" for m, ok in v["clean"].items()) + " |")
    first = next(iter(out["scenarios"].values()))
    L += ["", "| Scenario | " + " | ".join(i["item"] for i in first["items"]) + " | Total (×1.2, h) | Capacity (h) | Cuts |",
          "|---" * (len(first["items"]) + 4) + "|"]
    for name, e in out["scenarios"].items():
        L.append(f"| {name} | " + " | ".join(f"{i['with_contingency_h']:.1f}" for i in e["items"]) +
                 f" | {e['total_before_cuts_h']:.1f} | {e['capacity_h']} | " +
                 (", ".join(f"{a['item']} {a['to']}" for a in e["cuts_applied"]) or "none") +
                 ("" if e["fits"] else " — STILL DOES NOT FIT") + " |")
    L += ["", "Per-round times: mean over runs (s); prostate_ms values are derived from the measured prostate components "
          "(scaling in the JSON). Not-clean measurements (overlap with other work on this machine): " +
          "; ".join(f"{k}: {v}" for k, v in CONTAMINATED.items()) + ".",
          f"Capacity 'evenings': tonight {TONIGHT_H} h (assumed) + the 17 Oct full days + {EVENING_HOURS} h on other days "
          f"(assumed). Capacity 'overnight': tonight from the v2-screen end to midnight + full days unchanged + "
          f"{WEEKDAY_OVERNIGHT_H} h on other weekdays (20 h - 4 h buffer). To {DEADLINE}."]
    with open(os.path.join(OUT, "recost_18oct.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
