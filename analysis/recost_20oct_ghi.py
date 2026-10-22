"""
Recost with items g, g2, h, i inserted (user order 20 Oct 2026, evening): a (running) -> b -> g -> g2 -> h -> i -> c -> d
-> e -> f; g, g2, h, i never cut; the pre-registered cut order applies to c-f only. Overnight mode. Costs from
analysis/recost_20oct.py (measured components); additionally:
  g  prostate_ms_runmc ~ prostate_ms costs (derived: client slices 1446 vs 1483, fitness 117 vs 106, test 190 vs 173)
  h  follow-up (conditional on the screen): XFedGA x 5 seeds on prostate_ms with test, counted as if needed
  remaining part of item a: its queue status (runs done) and the per-run cost.
Capacity from NOW to 27 Oct 24:00. Writes results/timing/recost_20oct_ghi.{json,md}.
    & $py analysis/recost_20oct_ghi.py
"""
import datetime as dt
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recost_20oct import CONT, capacity, costs, items, run_h  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(REPO, "results", "runs", "_queues")
CUTS = [("fedprox_bounds", 0.0), ("parity_c", 0.0), ("ablations", 2 / 3), ("h_rob2", 0.0), ("fedga_paper", 0.0),
        ("ablations", 0.0)]   # c-f only (dose-response p = 0.25 is item a, already running: not cut)


def main():
    r, test, derived, _ = costs()
    r["prostate_ms_runmc"] = dict(r["prostate_ms"])
    test["prostate_ms_runmc"] = test["prostate_ms"] * 190 / 173
    base = {k: v for k, _, v in items(r, test)}
    qa = yaml.safe_load(open(os.path.join(REPO, "queues", "post_a_dose_response.yaml"), encoding="utf-8"))["runs"]
    sa = os.path.join(Q, "post_a_dose_response_status.json")
    done = len(json.load(open(sa, encoding="utf-8"))["runs"]) if os.path.exists(sa) else 0
    a_left = sum(run_h(r, test, e["task"], e["method"], 1) for e in qa[done:])
    it = [("a_rest", f"a. dose-response, remaining {len(qa) - done} of {len(qa)} runs", a_left),
          ("b", "b. Flower demo + parity", base["flower_demo"] + base["parity"]),
          ("g", "g. prostate_ms_runmc: FedGA, XFedGA+ (DSC-only), XFedGA+, XFedGA x 5 seeds (derived from prostate_ms)",
           sum(run_h(r, test, "prostate_ms_runmc", m, 5) for m in ["fedga", "xfedga_plus_dsc", "xfedga_plus", "xfedga"])),
          ("g2", "g2. breast FedGA + XFedGA+ (DSC-only) x seeds 6-10",
           sum(run_h(r, test, "breast", m, 5) for m in ["fedga", "xfedga_plus_dsc"])),
          ("h", "h. fitness-weight screen: 3 combinations x 2 seeds, no test",
           3 * 2 * (r["prostate_ms"]["_setup"] + 15 * r["prostate_ms"]["xfedga"]) / 3600),
          ("h_followup", "h. conditional follow-up: 5 seeds with test (counted as if needed)",
           run_h(r, test, "prostate_ms", "xfedga", 5)),
          ("i", "i. beta screen: 3 betas x 2 seeds, no test",
           3 * 2 * (r["prostate_ms"]["_setup"] + 15 * r["prostate_ms"]["xfedga"]) / 3600),
          ("fedga_paper", "c. FedGA paper GA settings", base["fedga_paper"]),
          ("h_rob2", "d. H-ROB2", base["h_rob2"]),
          ("ablations", "e. ablations", base["ablations"]),
          ("fedprox_bounds", "f. FedProx + bounds", base["fedprox_bounds"])]
    # 'parity_c' is a placeholder: parity runs inside item b, which is never cut here (b runs before g)
    cap = capacity(dt.datetime.now())
    orig = {k: v * CONT for k, _, v in it}
    h = dict(orig)
    tot = sum(h.values())
    applied = []
    for k, keep in CUTS:
        if tot <= cap or k not in h:
            continue
        new = orig[k] * keep
        if h[k] == 0 or new >= h[k]:
            continue
        applied.append({"item": k, "to": "dropped" if keep == 0 else f"x{keep:.2f}"})
        tot -= h[k] - new
        h[k] = new
    out = {"capacity_from": str(dt.datetime.now()), "capacity_h": cap, "items": [
        {"item": k, "label": lab, "h_with_contingency": orig[k], "after_cuts": h[k]} for k, lab, _ in it],
        "total_h": tot, "cuts_applied": applied, "fits": tot <= cap, "derived": derived}
    json.dump(out, open(os.path.join(REPO, "results", "timing", "recost_20oct_ghi.json"), "w", encoding="utf-8"),
              indent=1, default=str)
    L = [f"Capacity from now ({dt.datetime.now():%a %d %b %H:%M}) to 27 Oct 24:00, overnight mode: {cap} GPU-h.", "",
         "| Item | Hours (×1.2) | After cuts |", "|---|---|---|"] + \
        [f"| {lab} | {orig[k]:.1f} | {h[k]:.1f} |" for k, lab, _ in it] + \
        ["", f"Total: {tot:.1f} h of {cap} h → {'fits' if tot <= cap else 'DOES NOT FIT'}. Cuts (c–f only): "
         + (", ".join(f"{a['item']} {a['to']}" for a in applied) or "none") + ".",
         "g is costed with the measured prostate_ms components (derived); the XAI metrics pass is not included."]
    open(os.path.join(REPO, "results", "timing", "recost_20oct_ghi.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
