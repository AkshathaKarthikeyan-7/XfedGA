"""
Queue of the combined validation-only screen (pre-registered 18 Oct 2026: prostate_ms and XFedGA+ pre-registrations):
  prostate_ms: FedGA, XFedGA, XFedGA+, XFedGA+ (DSC-only) x protocols v1 / v2 x seeds 1-2 (16 runs; tags msv1screen /
               msv2screen; v2 = 2 local epochs, fitness subsample 120 (capped at the fitness set), FOV crop 110 mm +
               z-score; S from results/audit/fov_crop_prostate_ms.json);
  breast:      XFedGA+ and XFedGA+ (DSC-only) x seeds 1-2 under the ADOPTED breast protocol (results/protocol_v2/
               v2_adoption.json), tag xplusscreen (FedGA / XFedGA cells reused).
Order: prostate_ms seed-1 cells, breast seed 1, prostate_ms seed-2 cells, breast seed 2. No test evaluation.
Logging (evaluation only): log_wagg_val, log_client_val. Writes queues/screen_19oct.yaml.
    & $py analysis/make_screen_queue.py
"""
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = ["fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]
LOG = ["evaluation.log_wagg_val=true", "evaluation.log_client_val=true"]
V2_COMMON = ["fitness.subsample=120", "train.local_epochs=2"]


def main():
    dec = json.load(open(os.path.join(REPO, "results", "protocol_v2", "v2_adoption.json"), encoding="utf-8"))["decision"]
    assert dec["breast"]["complete"], "breast v2 decision incomplete"
    breast_v2 = dec["breast"]["adopt_v2"]
    side = json.load(open(os.path.join(REPO, "results", "audit", "fov_crop_prostate_ms.json"), encoding="utf-8"))["side_mm"]
    ms = {"v1": LOG, "v2": V2_COMMON + [f"preprocessing.fov_crop_mm={side}", "preprocessing.intensity=zscore"] + LOG}
    breast_set = (V2_COMMON if breast_v2 else []) + LOG
    runs = []
    for seed in (1, 2):
        for proto in ("v1", "v2"):
            runs += [{"task": "prostate_ms", "method": m, "seed": seed, "set": ms[proto], "no_test": True,
                      "tag": f"ms{proto}screen"} for m in METHODS]
        runs += [{"task": "breast", "method": m, "seed": seed, "set": breast_set, "no_test": True, "tag": "xplusscreen"}
                 for m in ["xfedga_plus", "xfedga_plus_dsc"]]
    q = {"name": "screen_19oct", "out": "results/runs",
         "note": f"Combined validation-only screen (prostate_ms protocol v1/v2 + XFedGA+; breast XFedGA+ under breast "
                 f"{'v2' if breast_v2 else 'v1'}). Pre-registered in plan.md; no test evaluation.", "runs": runs}
    path = os.path.join(REPO, "queues", "screen_19oct.yaml")
    yaml.safe_dump(q, open(path, "w", encoding="utf-8"), sort_keys=False, width=200)
    print(f"{len(runs)} runs -> {path}; breast protocol {'v2' if breast_v2 else 'v1'}; prostate_ms S = {side} mm")


if __name__ == "__main__":
    main()
