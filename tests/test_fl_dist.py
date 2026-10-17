"""
fl_dist (Flower, separate OS processes over loopback gRPC) must reproduce the custom loop: a 1-round
tiny XFedGA run with the regression seed matches tests/golden/regression_prostate.json bit-for-bit,
and the message audit shows only weights, n_i, q_i and the client index. Slow (3 processes, GPU).
"""
import csv
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import REPO, require_path
from tests.test_regression import GOLDEN, TINY


@pytest.mark.data
@pytest.mark.slow
def test_fl_dist_matches_custom_loop_and_passes_audit(tmp_path):
    require_path("raw data/PROMISE12")
    cmd = [sys.executable, os.path.join(REPO, "fl_dist", "launch.py"), "--task", "prostate", "--method", "xfedga",
           "--seed", "7", "--tag", "fltest", "--out", str(tmp_path), "--no_test", "--set", *TINY]
    res = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1200)
    assert res.returncode == 0, res.stdout[-2000:]
    run = next(p for p in tmp_path.iterdir() if p.is_dir())
    row = next(csv.DictReader(open(run / "round_log.csv", encoding="utf-8")))
    golden = json.load(open(GOLDEN, encoding="utf-8"))["xfedga"]["round"]
    for k in ["val_dsc", "val_hd95_norm", "val_hd95_px", "val_ec", "val_pg"]:
        assert float(row[k]) == golden[k], k
    for k in ["q_i", "alpha_i", "ga_best_per_gen", "ga_mean_per_gen"]:
        assert row[k] == golden[k], k
    audit = json.load(open(run / "message_audit.json", encoding="utf-8"))
    assert set(audit["metric_keys_seen"]) <= {"q_i", "client_index"}
    assert audit["all_shapes_ok"] and audit["dtypes_seen"] == ["float32"] and audit["config_keys_sent"] == ["round"]
    assert audit["messages_received"] == 2
