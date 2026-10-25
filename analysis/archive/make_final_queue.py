"""
FINAL HEADLINE queue (pre-registered: plan.md prostate_ms / XFedGA+ / acceptance-check pre-registrations). Written
only after all screen decisions exist; test evaluated once per run.

  prostate_ms (adopted protocol): FedAvg, FedGA (baseline), XFedGA x seeds 1-5
                                  + XFedGA+ and XFedGA+ (DSC-only) x 5 if XFedGA+ adopted
                                  + acceptance-check arm (of the selected method) x 5 if the acceptance check adopted
  breast (adopted protocol; v1 -> the v1 headline FedAvg / FedGA / XFedGA are REUSED, not rerun):
                                  only the adopted arms x 5 seeds (if breast v2 were adopted, the three primary arms
                                  would also be run)
Order: by seed (all arms of seed 1, then seed 2, ...), prostate_ms before breast within a seed. Tag "final".
Logging (evaluation only): log_wagg_val, log_client_val. Cuts: none (results/timing/recost_18oct.md fits).
Writes queues/final_headline.yaml and results/final/final_queue_record.json.
    & $py analysis/make_final_queue.py
"""
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # analysis/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # analysis/archive/
from protocol_sets import adopted_protocols, overrides, xplus_adopted  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    protos, xplus = adopted_protocols(), xplus_adopted()
    acc = json.load(open(os.path.join(REPO, "results", "acceptance", "acceptance_adoption.json"), encoding="utf-8"))
    if acc["outcome"].startswith("undecided"):
        raise RuntimeError("acceptance decision incomplete")
    sel = acc["selected_method"]
    extra = (["xfedga_plus", "xfedga_plus_dsc"] if xplus else []) + ([f"{sel}+acceptance"] if acc["adopt_acceptance"] else [])
    arms = {"prostate_ms": ["fedavg", "fedga", "xfedga"] + extra,
            "breast": (["fedavg", "fedga", "xfedga"] if protos["breast"] == "v2" else []) + extra}
    runs = []
    for seed in range(1, 6):
        for task in ("prostate_ms", "breast"):
            for arm in arms[task]:
                method, a = (arm.split("+")[0], True) if arm.endswith("+acceptance") else (arm, False)
                runs.append({"task": task, "method": method, "seed": seed,
                             "set": overrides(task, protos[task]) + (["ga.acceptance_check=true"] if a else []),
                             "tag": "finalacc" if a else "final"})
    q = {"name": "final_headline", "out": "results/runs",
         "note": f"FINAL HEADLINE (test evaluated once per run). Protocols {protos}; XFedGA+ adopted: {xplus}; "
                 f"acceptance adopted: {acc['adopt_acceptance']} (selected method {sel}). Breast primary arms reused "
                 f"from the v1 headline: {protos['breast'] == 'v1'}.", "runs": runs}
    yaml.safe_dump(q, open(os.path.join(REPO, "queues", "final_headline.yaml"), "w", encoding="utf-8"),
                   sort_keys=False, width=200)
    os.makedirs(os.path.join(REPO, "results", "final"), exist_ok=True)
    rec = {"protocols": protos, "xfedga_plus_adopted": xplus, "acceptance_adopted": acc["adopt_acceptance"],
           "selected_method": sel, "arms": arms, "n_runs": len(runs),
           "breast_primary_arms_reused_from": "v1 headline (commit 49bf2ad)" if protos["breast"] == "v1" else None}
    json.dump(rec, open(os.path.join(REPO, "results", "final", "final_queue_record.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
