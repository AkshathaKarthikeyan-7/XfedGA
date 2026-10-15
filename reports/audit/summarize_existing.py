"""
Audit: summarize the pre-existing (superseded, OOD-selected) results recorded
in the root-level round_history_*.csv files. Read-only; writes
reports/audit/existing_results.json.

All numbers here are slice/image-level means on the OOD half of the server
split, logged during training; EC and pointing game use only the first 30
OOD samples (evaluate_explainability, max_samples=30).

Usage (from repo root):
    python reports/audit/summarize_existing.py
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FILES = {
    "breast_fedavg": "round_history_fedavg_baseline.csv",
    "breast_base_fedga": "round_history_base_fedga.csv",
    "breast_xfedga": "round_history_xfedga_fedga.csv",
    "prostate_base_fedga": "round_history_base_fedga_prostate.csv",
    "prostate_xfedga": "round_history_xfedga_fedga_prostate.csv",
    "nested_breast_fedavg": "xfedga/round_history_fedavg_baseline.csv",
    "nested_breast_base_fedga": "xfedga/round_history_base_fedga.csv",
    "nested_breast_xfedga": "xfedga/round_history_xfedga_fedga.csv",
}


def main():
    out = {}
    for key, rel in FILES.items():
        rows = list(csv.DictReader(open(os.path.join(REPO, rel))))
        dsc = np.array([float(r["ood_dsc"]) for r in rows])
        rec = {"file": rel, "n_rounds": len(rows),
               "best_round_by_ood_dsc": int(np.argmax(dsc)) + 1, "best_ood_dsc": float(dsc.max()),
               "final_round_ood_dsc": float(dsc[-1]),
               "last5_mean_ood_dsc": float(dsc[-5:].mean()), "last5_std_ood_dsc": float(dsc[-5:].std())}
        for col in ["ood_ec", "ood_pointing_game"]:
            if col in rows[0]:
                v = np.array([float(r[col]) for r in rows])
                rec[f"{col}_at_best_round"] = float(v[np.argmax(dsc)])
                rec[f"last5_mean_{col}"] = float(v[-5:].mean())
        if "client_saliency_scores" in rows[0] and rows[0]["client_saliency_scores"]:
            q = np.array([[float(x) for x in r["client_saliency_scores"].split(",")] for r in rows])
            rec["q_i_min_per_client"] = q.min(0).tolist()
            rec["q_i_max_per_client"] = q.max(0).tolist()
            rec["q_i_round_to_round_absdiff_mean_per_client"] = np.abs(np.diff(q, axis=0)).mean(0).tolist()
        out[key] = rec
    with open(os.path.join(HERE, "existing_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
