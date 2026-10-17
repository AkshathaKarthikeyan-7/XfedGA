"""End-to-end check of analysis/make_tables.py on synthetic run folders (no real results touched)."""
import csv
import json
import os
import subprocess
import sys

import numpy as np

from tests.conftest import REPO

ROUND_COLS = ["round", "val_dsc", "val_hd95_norm", "val_hd95_px", "val_ec", "val_pg", "q_i", "alpha_i",
              "ga_best_per_gen", "ga_mean_per_gen", "fitness_evals", "local_s", "qi_s", "ga_s", "val_s", "round_s",
              "peak_reserved_MiB"]


def fake_run(root, task, method, seed, shift, rng, tag="", cam=("fullres", "pred")):
    rid = f"{task}_{method}_seed{seed}_abcdef12" + (f"_{tag}" if tag else "")
    d = os.path.join(root, rid)
    os.makedirs(d)
    with open(os.path.join(d, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        f.write(f"fitness:\n  cam_layer: {cam[0]}\n  cam_target: {cam[1]}\n")
    q = "[0.2, 0.3]" if method == "xfedga" else "[]"
    with open(os.path.join(d, "round_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROUND_COLS)
        w.writeheader()
        for t in range(1, 6):
            w.writerow({"round": t, "val_dsc": 0.5 + shift + 0.01 * t, "val_hd95_norm": 0.2, "val_hd95_px": 50, "val_ec": 0.1,
                        "val_pg": 0.5, "q_i": q, "alpha_i": "[0.4, 0.6]", "ga_best_per_gen": "[]", "ga_mean_per_gen": "[]",
                        "fitness_evals": 21, "local_s": 40, "qi_s": 2, "ga_s": 50 if method != "fedavg" else 0, "val_s": 5,
                        "round_s": 100 if method == "xfedga" else 60, "peak_reserved_MiB": 700})
    with open(os.path.join(d, "test_per_image.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "group_id", "dsc", "hd95_norm", "hd95_px", "ec", "pg"])
        for c in range(20):
            base = 0.6 + 0.05 * np.sin(c)
            for k in range(3):
                w.writerow([f"c{c}#{k}", f"c{c}", base + shift + rng.normal(0, 0.005), 0.2 - shift, 40, 0.1, 1.0])
    json.dump({"run_id": rid, "task": task, "method": method, "seed": seed, "selected_round": 4,
               "smoothed_val_dsc": [0.5, 0.55, 0.56, 0.57, 0.56], "test_evaluated": True},
              open(os.path.join(d, "run_meta.json"), "w"))


def test_tables_comparisons_outcome_and_latex(tmp_path):
    rng = np.random.default_rng(0)
    runs = tmp_path / "runs"
    for seed in (1, 2, 3):
        fake_run(runs, "prostate", "fedavg", seed, 0.00, rng)
        fake_run(runs, "prostate", "fedga", seed, 0.00, rng)
        fake_run(runs, "prostate", "xfedga", seed, 0.05, rng)
        fake_run(runs, "prostate", "xfedga", seed, -0.30, rng, tag="hetero")    # other group: must not mix in
        fake_run(runs, "prostate", "fedavg", seed + 10, -0.30, rng, cam=("halfres", "all"))   # old CAM: excluded
    out = tmp_path / "tables"
    res = subprocess.run([sys.executable, os.path.join(REPO, "analysis", "make_tables.py"), "--runs", str(runs / "*"),
                          "--group", "main", "--n_boot", "2000", "--out", str(out)], capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, res.stderr[-2000:]
    data = json.load(open(out / "tables.json"))
    assert all("hetero" not in r for r in data["runs"]) and len(data["runs"]) == 9   # old-CAM runs excluded
    c = next(x for x in data["comparisons"] if x["method"] == "xfedga" and x["comparator"] == "fedga" and x["metric"] == "dsc")
    assert c["n_cases"] == 20 and abs(c["mean_diff"] - 0.05) < 0.005 and c["outcome"] == "WIN"
    h = next(x for x in data["comparisons"] if x["method"] == "fedga" and x["comparator"] == "fedavg" and x["metric"] == "dsc")
    assert h["outcome"] == "TIE"
    tex = (out / "comparisons.tex").read_text(encoding="utf-8")
    assert tex.startswith(r"\begin{tabular}") and r"\hline" in tex and "booktabs" not in tex and "toprule" not in tex
    assert "WIN" in (out / "outcome.md").read_text(encoding="utf-8")
    md = (out / "comparisons.md").read_text(encoding="utf-8")
    assert "PRIMARY" in md and "secondary" in md and "Wilcoxon p" in md
    prim = [line for line in md.splitlines() if "PRIMARY" in line and "| dsc |" in line]
    assert len(prim) == 1 and "XFedGA vs FedGA (baseline)" in prim[0]
    assert (out / "fig_val_dsc_prostate.png").exists() and (out / "fig_qi_alpha_prostate.png").exists()
