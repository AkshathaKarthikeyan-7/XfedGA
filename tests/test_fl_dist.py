"""
fl_dist (Flower, separate OS processes over loopback gRPC) must reproduce the custom loop: a 1-round
tiny XFedGA run with the regression seed matches tests/golden/regression_prostate.json bit-for-bit,
and the message audit shows exactly weights, n_i, q_i and the client index. Slow (3 processes, GPU).
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
    # exactly what crosses the wire (ARCHITECTURE.md, "What fl_dist sends"):
    #   client -> server: the model weights (63 float32 arrays), n_i (num_examples), q_i, client_index
    #   server -> client: the global weights and {"round"}
    audit = json.load(open(run / "message_audit.json", encoding="utf-8"))
    assert audit["metric_keys_seen"] == ["client_index", "q_i"]
    assert audit["all_shapes_ok"] and audit["dtypes_seen"] == ["float32"]
    assert audit["config_keys_sent"] == ["round"]
    received = audit["detail"]["received"]
    assert len(received) == 2 and all(r["n_arrays"] == 63 for r in received)
    assert sorted(r["num_examples"] for r in received) == [379, 399]          # n_i of the two PROMISE12 clients
    assert sorted(r["client_index"] for r in received) == [0, 1]
    assert all(r["metric_keys"] == ["client_index", "q_i"] for r in received)
