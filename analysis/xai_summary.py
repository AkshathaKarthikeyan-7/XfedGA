"""
Per task and method: mean ± s.d. over seeds of each independent XAI metric on the test checkpoints
(per-run value = mean over test images, results/xai/<run_id>/summary.json from analysis/xai_metrics.py).
Context for the H-XAI comparisons in results/tables/headline (e.g. ceiling effects). Headline runs only.
Writes results/xai/summary_headline.md / .json.
    & $py analysis/xai_summary.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XAI = os.path.join(REPO, "results", "xai")
METRICS = ["pg_gradcampp", "rho_after_all_layers", "pg_gradcam", "iou_gradcampp", "deletion_auc", "insertion_auc"]
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA"}


def main():
    cells = {}
    for p in glob.glob(os.path.join(XAI, "*", "summary.json")):
        rid = os.path.basename(os.path.dirname(p))
        m = re.fullmatch(r"(prostate|breast)_(fedavg|fedga|xfedga)_seed[1-5]_[0-9a-f]{8}", rid)
        if not m:
            continue
        s = json.load(open(p, encoding="utf-8"))
        assert s["split"] == "test" and s["cam_layer"] == "fullres" and s["cam_target"] == "pred", rid
        cells.setdefault((m.group(1), m.group(2)), []).append(s)
    out, lines = {}, ["| Task | Method | Seeds | " + " | ".join(METRICS) + " |", "|---" * (3 + len(METRICS)) + "|"]
    for task in ["prostate", "breast"]:
        for meth in ["fedavg", "fedga", "xfedga"]:
            rs = cells.get((task, meth), [])
            if not rs:
                continue
            st = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs], ddof=1))) for k in METRICS}
            out[f"{task}/{meth}"] = {"seeds": len(rs), **{k: {"mean": v[0], "sd": v[1]} for k, v in st.items()}}
            lines.append(f"| {task} | {LABEL[meth]} | {len(rs)} | " +
                         " | ".join(f"{v[0]:.3f} ± {v[1]:.3f}" for v in st.values()) + " |")
    lines += ["", "Test checkpoints (validation-selected), mean ± s.d. over seeds of the per-run mean over test images. "
              "rho_after_all_layers: lower = more faithful. Deletion / insertion AUC are exploratory (context caveat)."]
    with open(os.path.join(XAI, "summary_headline.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(XAI, "summary_headline.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
