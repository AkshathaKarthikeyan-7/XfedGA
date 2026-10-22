"""
Recost on Thu 22 Oct 2026 after the Flower hang (GPU idle about Wed 21:15 to Thu 07:00). Remaining queue (user order
22 Oct): suite -> g -> d (remaining 3 runs) -> b (Flower, 6 runs, seed 1 = demo) -> e -> f. Never cut: g, the Flower demo
run (seed 1), the d rerun. The pre-registered cut order applies to e / f only: f dropped first, then e to 2 seeds,
then e dropped (reported as a limitation). Costs: analysis/recost_20oct.py (measured components; derived costs
labelled there); g ~ prostate_ms (derived); the suite 0.3 h; XAI on g 20 runs x 3 min.
Capacity: overnight mode, from now to 27 Oct 24:00. Writes results/timing/recost_22oct.{json,md}.
    & $py analysis/recost_22oct.py
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recost_20oct import CONT, capacity, costs, items, run_h  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    r, test, _, _ = costs()
    r["prostate_ms_runmc"] = dict(r["prostate_ms"])
    test["prostate_ms_runmc"] = test["prostate_ms"] * 190 / 173
    base = {k: v for k, _, v in items(r, test)}
    it = [("suite", "full suite before the g code commit", 0.3),
          ("g", "g: prostate_ms_runmc, 4 methods x 5 seeds (derived from prostate_ms)",
           sum(run_h(r, test, "prostate_ms_runmc", m, 5) for m in ["fedga", "xfedga_plus_dsc", "xfedga_plus", "xfedga"])),
          ("xai_g", "XAI metrics on the 20 g runs", 20 * 3 / 60),
          ("d_rest", "d: remaining 3 runs (XFedGA+ seed 2, XFedGA and XFedGA+ seed 3)",
           run_h(r, test, "breast", "xfedga_plus", 2) + run_h(r, test, "breast", "xfedga", 1)),
          ("b_demo", "b: Flower seed 1 (demo: XFedGA, XFedGA+)",
           sum(run_h(r, test, "breast", m, 1, factor=1.15) for m in ["xfedga", "xfedga_plus"])),
          ("b_parity", "b: Flower seeds 2-3 (parity)", base["parity"]),
          ("e", "e: ablations, 2 tasks x 3 seeds", base["ablations"]),
          ("f", "f: FedProx + bounds", base["fedprox_bounds"])]
    cap = capacity(dt.datetime.now())
    orig = {k: v * CONT for k, _, v in it}
    h = dict(orig)
    tot = sum(h.values())
    applied = []
    for k, keep in [("f", 0.0), ("e", 2 / 3), ("e", 0.0)]:
        if tot <= cap:
            break
        new = orig[k] * keep
        if new >= h[k]:
            continue
        applied.append({"item": k, "to": "dropped" if keep == 0 else f"x{keep:.2f}"})
        tot -= h[k] - new
        h[k] = new
    out = {"capacity_h": cap, "items": [{"item": k, "label": lab, "h": orig[k], "after_cuts": h[k]} for k, lab, _ in it],
           "total_h": tot, "cuts": applied, "fits": tot <= cap, "computed": str(dt.datetime.now())}
    json.dump(out, open(os.path.join(REPO, "results", "timing", "recost_22oct.json"), "w", encoding="utf-8"), indent=1)
    acc, eta = dt.datetime.now(), {}
    for k, lab, v in it:                       # wall-clock ETA assuming continuous GPU use, without contingency
        acc += dt.timedelta(hours=v if h[k] > 0 else 0)
        eta[k] = acc
    L = [f"Capacity from {dt.datetime.now():%a %d %b %H:%M} to 27 Oct 24:00 (overnight mode): {cap} GPU-h.", "",
         "| Item | Hours (×1.2) | After cuts | Wall-clock ETA (no contingency) |", "|---|---|---|---|"] + \
        [f"| {lab} | {orig[k]:.1f} | {h[k]:.1f} | {eta[k]:%a %d %b %H:%M} |" for k, lab, _ in it] + \
        ["", f"Total {tot:.1f} h of {cap} h → {'fits' if tot <= cap else 'DOES NOT FIT'}; cuts: "
         + (", ".join(f"{a['item']} {a['to']}" for a in applied) or "none") + "."]
    open(os.path.join(REPO, "results", "timing", "recost_22oct.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
