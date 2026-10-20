"""
Queues of the post-headline items a-f (user order 19 Oct 2026; each pre-registered in plan.md in its own commit
before any of its runs). Adopted protocols: breast v1, prostate_ms v2 (results/protocol_v2/v2_adoption.json).
Overrides per (task, protocol) from analysis/protocol_sets.py, so every item uses the settings of its comparator runs.
  a dose_response : breast, BUS-UCLM (client 1) corrupted at p = 0.5, 1.0 first, then 0.25; FedAvg, FedGA, XFedGA,
                    XFedGA+; seeds 1-3; tags dose_p050 / dose_p100 / dose_p025 (p = 0 reused)
  b flower        : fl_dist, breast, XFedGA and XFedGA+, seed 1 (demo) then seeds 2-3 (parity); tag fl
                    (XFedGA without logging flags, as its comparator = the v1 headline; XFedGA+ with them, as the final runs)
  c paperga       : FedGA (paper GA settings: population 10, generations 20, rounds 20), both tasks, seeds 1-3
  d hrob2         : breast, BUS-UCLM always reports q_i = 1; XFedGA, XFedGA+; seeds 1-3
  e ablations     : B, C, D, F, random search; both tasks; seeds 1-3; tag abl
  f fedprox_bounds: FedProx x seeds 1-5, centralized and local-only x seeds 1-3; both tasks
Writes queues/post_<item>.yaml.
    & $py analysis/make_post_queues.py
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol_sets import adopted_protocols, overrides  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ["prostate_ms", "breast"]


def q(name, note, runs):
    path = os.path.join(REPO, "queues", f"post_{name}.yaml")
    yaml.safe_dump({"name": f"post_{name}", "out": "results/runs", "note": note, "runs": runs}, open(path, "w", encoding="utf-8"),
                   sort_keys=False, width=200)
    print(f"{name}: {len(runs)} runs")


def main():
    P = adopted_protocols()
    assert P == {"breast": "v1", "prostate_ms": "v2"}, P
    dose = []
    for p, tag in [(0.5, "dose_p050"), (1.0, "dose_p100"), (0.25, "dose_p025")]:
        for s in (1, 2, 3):
            for m in ["fedavg", "fedga", "xfedga", "xfedga_plus"]:
                dose.append({"task": "breast", "method": m, "seed": s, "tag": tag,
                             "set": overrides("breast", "v1") + ["scenario.corrupted_label_client=1",
                                                                 f"scenario.corrupted_label_fraction={p}"]})
    q("a_dose_response", "a. Dose-response (pre-registered, plan.md). Test evaluated once per run.", dose)
    fl = []
    for s in (1, 2, 3):
        for m in ["xfedga", "xfedga_plus"]:
            fl.append({"task": "breast", "method": m, "seed": s, "tag": "fl", "runner": "fl_dist",
                       "set": overrides("breast", "v1", log=(m == "xfedga_plus"))})
    q("b_flower", "b. Flower demo (seed 1) + parity (seeds 2-3), fl_dist, breast (pre-registered, plan.md).", fl)
    pga = [{"task": t, "method": "fedga", "seed": s, "tag": "paperga",
            "set": overrides(t, P[t]) + ["ga.population=10", "ga.generations=20", "train.rounds=20"]}
           for s in (1, 2, 3) for t in TASKS]
    q("c_paperga", "c. FedGA (paper GA settings), declared batch deviation (pre-registered, plan.md).", pga)
    hr = [{"task": "breast", "method": m, "seed": s, "tag": "hrob2",
           "set": overrides("breast", "v1") + ["scenario.noisy_qi_client=1"]}
          for s in (1, 2, 3) for m in ["xfedga", "xfedga_plus"]]
    q("d_hrob2", "d. H-ROB2 noisy-q_i client (pre-registered 661a173 + plan.md).", hr)
    ab = [{"task": t, "method": m, "seed": s, "tag": "abl", "set": overrides(t, P[t])}
          for s in (1, 2, 3) for t in TASKS for m in ["abl_B", "abl_C", "abl_D", "abl_F", "random_search"]]
    q("e_ablations", "e. Ablations B, C, D, F + random search (pre-registered, plan.md).", ab)
    fb = [{"task": t, "method": "fedprox", "seed": s, "tag": "fedprox", "set": overrides(t, P[t])}
          for s in (1, 2, 3, 4, 5) for t in TASKS] + \
         [{"task": t, "method": m, "seed": s, "tag": "bounds", "set": overrides(t, P[t])}
          for s in (1, 2, 3) for t in TASKS for m in ["centralized", "local_only"]]
    q("f_fedprox_bounds", "f. FedProx and the centralized / local-only bounds (pre-registered, plan.md).", fb)


if __name__ == "__main__":
    main()
