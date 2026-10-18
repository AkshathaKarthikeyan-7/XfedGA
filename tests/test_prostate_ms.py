"""
Multi-site prostate task (prostate_ms; decision B of 18 Oct 2026): sites as clients, BMC as held-out server site.
Fast tests: the server split rule (8 / 8 / 14 in gland-size blocks with a remainder rule) and the data plumbing.
Slow tests (GPU): a 1-round 5-client run in the custom loop and in fl_dist.
"""
import os

import numpy as np
import pytest

from data.splits import decathlon_case_split
from experiments.config import load_config
from tests.conftest import require_path

MS = "raw data/MultiSiteProstate"


def _cases(n):
    return [(f"case{i:02d}", "x") for i in range(n)], {f"case{i:02d}": i + 1 for i in range(n)}


def test_remainder_roles_give_8_8_14_and_keep_blocks():
    cases, sizes = _cases(30)
    s = decathlon_case_split(cases, sizes, 42, ("fitness", "validation", "test", "test"), ("fitness", "validation"))
    assert [len(s[r]) for r in ("fitness", "validation", "test")] == [8, 8, 14]
    assert sorted(sum(s.values(), [])) == sorted(cases)                       # disjoint and complete
    top2 = {c for c, _ in sorted(cases, key=lambda c: sizes[c[0]])[-2:]}      # the 2 largest glands
    assert {c for c, _ in s["test"]}.isdisjoint(top2)
    with pytest.raises(ValueError):
        decathlon_case_split(cases, sizes, 42)                                 # remainder without a rule


def test_divisible_split_is_unchanged_by_the_remainder_option():
    cases, sizes = _cases(32)
    a = decathlon_case_split(cases, sizes, 42)
    b = decathlon_case_split(cases, sizes, 42, remainder_roles=("fitness", "validation"))
    assert a == b and [len(a[r]) for r in ("fitness", "validation", "test")] == [8, 8, 16]


def test_config_is_self_contained():
    cfg = load_config("prostate_ms")
    paths = [c["path"] for c in cfg["clients"]] + [cfg["server"]["path"]]
    assert [c["name"] for c in cfg["clients"]] == ["BIDMC", "HK", "I2CVB", "RUNMC", "UCL"] and cfg["server"]["name"] == "BMC"
    assert all(p.startswith(MS) for p in paths)
    assert not any("PROMISE12" in p or "Task05" in p for p in paths)
    load_config("prostate_ms", ["scenario.noisy_qi_client=4"])                # 5 clients: index 4 is valid
    with pytest.raises(ValueError):
        load_config("prostate_ms", ["scenario.noisy_qi_client=5"])


@pytest.mark.data
def test_prostate_ms_split_and_data():
    require_path(MS)
    from experiments.tasks import build_task, split_summary
    cfg = load_config("prostate_ms")
    data = build_task("prostate_ms", cfg)
    s = split_summary(data)
    assert list(s["clients"]) == ["BIDMC", "HK", "I2CVB", "RUNMC", "UCL"]
    assert [s["server"][r]["n_cases"] for r in ("fitness", "validation", "test")] == [8, 8, 14]
    roles = [set(s["server"][r]["cases"]) for r in ("fitness", "validation", "test")]
    assert not (roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
    for c in data.clients:                                        # every item comes from that site's folder
        assert all(os.path.normpath(p).startswith(os.path.normpath(os.path.join(MS, c.name)))
                   for p, _ in c.train.cases)
    x, y = data.clients[0].train[0]
    assert x.shape == (1, 384, 384) and set(np.unique(y.numpy())) <= {0, 1}
    assert len(data.fitness_subsample) == cfg["fitness"]["subsample"]
    assert build_task("prostate_ms", cfg).split_log["server"] == data.split_log["server"]   # deterministic


@pytest.mark.data
@pytest.mark.slow
def test_prostate_ms_one_round_custom_loop(tmp_path):
    require_path(MS)
    from experiments.run_federated import run_experiment
    from tests.test_switches import TINY
    res = run_experiment("prostate_ms", "xfedga", 5, TINY, out_root=str(tmp_path), evaluate_test=False)
    row = res["rounds"][0]
    assert len(res["meta"]["n_i"]) == 5
    import json
    assert len(json.loads(row["q_i"])) == len(json.loads(row["alpha_i"])) == 5
    assert 0.0 <= row["val_dsc"] <= 1.0


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="KNOWN LIMITATION 18 Oct 2026: 5 Flower client processes + the server do not fit "
                                       "on the 6 GB GPU together (CUDA OOM in round 1); fix: cross-process GPU lock")
def test_prostate_ms_one_round_fl_dist(tmp_path):
    require_path(MS)
    import subprocess
    import sys
    from tests.conftest import REPO
    from tests.test_switches import TINY
    out = str(tmp_path / "fl")
    r = subprocess.run([sys.executable, os.path.join(REPO, "fl_dist", "launch.py"), "--task", "prostate_ms",
                        "--method", "xfedga", "--seed", "5", "--no_test", "--out", out, "--set", *TINY],
                       capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    logs = [f for d, _, fs in os.walk(out) for f in fs if f.startswith("client_") and f.endswith(".log")]
    assert len(logs) == 5
