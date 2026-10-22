"""
Run overrides of each (task, protocol) exactly as used by the combined screen (analysis/make_screen_queue.py), so the
acceptance screen and the final headline use the same settings as the screen cells they are compared with.
  breast:      v1 = defaults; v2 = 2 local epochs + fitness subsample 120
  prostate_ms: v1 = defaults; v2 = 2 local epochs + fitness subsample 120 + FOV crop S (results/audit/
               fov_crop_<task>.json) + z-score
LOG (evaluation only, does not change a run): log_wagg_val, log_client_val.
"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = ["evaluation.log_wagg_val=true", "evaluation.log_client_val=true"]
V2_COMMON = ["fitness.subsample=120", "train.local_epochs=2"]


def overrides(task, proto, log=True):
    s = []
    if proto == "v2":
        s += V2_COMMON
        if task.startswith("prostate_ms"):      # each multi-site task: S from its own client masks
            side = json.load(open(os.path.join(REPO, "results", "audit", f"fov_crop_{task}.json"),
                                  encoding="utf-8"))["side_mm"]
            s += [f"preprocessing.fov_crop_mm={side}", "preprocessing.intensity=zscore"]
    elif proto != "v1":
        raise ValueError(proto)
    return s + (LOG if log else [])


def adopted_protocols():
    """task -> 'v1' | 'v2' from results/protocol_v2/v2_adoption.json; raises if a decision is incomplete."""
    d = json.load(open(os.path.join(REPO, "results", "protocol_v2", "v2_adoption.json"), encoding="utf-8"))["decision"]
    out = {}
    for t in ("breast", "prostate_ms"):
        if not d[t]["complete"]:
            raise RuntimeError(f"v2 decision incomplete for {t}")
        out[t] = "v2" if d[t]["adopt_v2"] else "v1"
    return out


def xplus_adopted():
    p = os.path.join(REPO, "results", "xfedga_plus", "xplus_adoption.json")
    d = json.load(open(p, encoding="utf-8"))
    if d["outcome"].startswith("undecided"):
        raise RuntimeError("XFedGA+ decision incomplete")
    return bool(d["adopt_xfedga_plus"])
