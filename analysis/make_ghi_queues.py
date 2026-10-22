"""
Queues of items g, g2, h, i (user order 20 Oct 2026; each pre-registered in plan.md in its own commit before any of
its runs). Overrides from analysis/protocol_sets.py (each multi-site task uses the FOV crop S of its own clients).
  g  site_runmc : prostate_ms_runmc (v2; S from results/audit/fov_crop_prostate_ms_runmc.json); FedGA (baseline),
                  XFedGA+ (DSC-only), XFedGA+, XFedGA; seeds 1-5; test once; tag runmc
  g2 replication: breast (v1); FedGA (baseline), XFedGA+ (DSC-only); NEW seeds 6-10; test once; tag rep
  h  fw screen  : prostate_ms (v2); XFedGA; seeds 1-2; NO test; fitness weights (1,0,0), (0.6,0.2,0.2), (0.4,0.4,0.2)
                  ((0.5,0.3,0.2) reused: the *_msv2screen XFedGA runs); tags fw100 / fw622 / fw442
  i  beta screen: prostate_ms (v2); XFedGA; seeds 1-2; NO test; beta 0, 0.2, 1.0 (0.5 reused: *_msv2screen XFedGA);
                  tags ms_betascreen_b0p0 / b0p2 / b1p0
Writes queues/post_<item>.yaml.
    & $py analysis/make_ghi_queues.py
    & $py analysis/make_ghi_queues.py --h_followup   # after the h decision: XFedGA x 5 seeds with test at the chosen
                                                     # weights, only if results/sensitivity/fw_screen_prostate_ms.json
                                                     # names follow_up_weights (declared secondary); exit code 3 if none
"""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_sets import overrides  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FW = {"fw100": [1.0, 0.0, 0.0], "fw622": [0.6, 0.2, 0.2], "fw442": [0.4, 0.4, 0.2]}
BETAS = {"ms_betascreen_b0p0": 0.0, "ms_betascreen_b0p2": 0.2, "ms_betascreen_b1p0": 1.0}


def q(name, note, runs):
    path = os.path.join(REPO, "queues", f"post_{name}.yaml")
    yaml.safe_dump({"name": f"post_{name}", "out": "results/runs", "note": note, "runs": runs},
                   open(path, "w", encoding="utf-8"), sort_keys=False, width=200)
    print(f"{name}: {len(runs)} runs")


def main():
    q("g_site_runmc", "g. Independent-site confirmation, prostate_ms_runmc (pre-registered, plan.md). Test once per run.",
      [{"task": "prostate_ms_runmc", "method": m, "seed": s, "tag": "runmc", "set": overrides("prostate_ms_runmc", "v2")}
       for s in (1, 2, 3, 4, 5) for m in ["fedga", "xfedga_plus_dsc", "xfedga_plus", "xfedga"]])
    q("g2_breast_replication", "g2. Breast replication, NEW seeds 6-10 (pre-registered, plan.md). Test once per run.",
      [{"task": "breast", "method": m, "seed": s, "tag": "rep", "set": overrides("breast", "v1")}
       for s in (6, 7, 8, 9, 10) for m in ["fedga", "xfedga_plus_dsc"]])
    q("h_fw_screen", "h. Fitness-weight sensitivity screen, prostate_ms, VALIDATION ONLY (pre-registered, plan.md).",
      [{"task": "prostate_ms", "method": "xfedga", "seed": s, "tag": tag, "no_test": True,
        "set": overrides("prostate_ms", "v2") + [f"fitness.weights=[{w[0]},{w[1]},{w[2]}]"]}
       for s in (1, 2) for tag, w in FW.items()])
    q("i_beta_screen", "i. Beta sensitivity screen, prostate_ms, VALIDATION ONLY (pre-registered, plan.md).",
      [{"task": "prostate_ms", "method": "xfedga", "seed": s, "tag": tag, "no_test": True,
        "set": overrides("prostate_ms", "v2") + [f"aggregation.beta={b}"]}
       for s in (1, 2) for tag, b in BETAS.items()])


def h_followup():
    d = json.load(open(os.path.join(REPO, "results", "sensitivity", "fw_screen_prostate_ms.json"), encoding="utf-8"))
    if d["outcome"].startswith("undecided"):
        raise RuntimeError("h screen incomplete")
    w = d["follow_up_weights"]
    if not w:
        print("insensitive: no follow-up")
        sys.exit(3)
    tag = "fwfinal" + "".join(str(int(round(10 * x))) for x in w)
    q("h_followup", f"h follow-up (declared secondary): XFedGA at weights {w}, prostate_ms v2, 5 seeds, test once.",
      [{"task": "prostate_ms", "method": "xfedga", "seed": s, "tag": tag,
        "set": overrides("prostate_ms", "v2") + [f"fitness.weights=[{w[0]},{w[1]},{w[2]}]"]} for s in (1, 2, 3, 4, 5)])


if __name__ == "__main__":
    h_followup() if "--h_followup" in sys.argv else main()
