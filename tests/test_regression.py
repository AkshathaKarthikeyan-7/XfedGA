"""
Regression pin: FedAvg, FedGA (baseline) and XFedGA must produce bit-identical outputs for a fixed
seed and a tiny config, before and after any code change (Session 3a adds method switches that
must leave these three methods untouched).

Golden values: tests/golden/regression_prostate.json. Regenerate ONLY for an intended change:
    $env:UPDATE_GOLDEN = "1"; & $py -m pytest tests/test_regression.py; Remove-Item Env:UPDATE_GOLDEN
"""
import hashlib
import json
import os

import pytest

from tests.conftest import REPO, require_path

GOLDEN = os.path.join(REPO, "tests", "golden", "regression_prostate.json")
TINY = ["train.rounds=1", "train.local_epochs=1", "ga.population=2", "ga.generations=1",
        "fitness.subsample=8", "saliency.subsample=4", "evaluation.explain_samples=8"]
KEYS = ["val_dsc", "val_hd95_norm", "val_hd95_px", "val_ec", "val_pg", "q_i", "alpha_i",
        "ga_best_per_gen", "ga_mean_per_gen", "fitness_evals"]


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprint(tmp_path, method):
    from experiments.run_federated import run_experiment
    res = run_experiment("prostate", method, 7, TINY, out_root=str(tmp_path / method), evaluate_test=False)
    return {"round": {k: res["rounds"][0][k] for k in KEYS}, "final_state_sha256": state_hash(res["final_state"]),
            "selected_round": res["meta"]["selected_round"], "n_i": res["meta"]["n_i"]}


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("method", ["fedavg", "fedga", "xfedga"])
def test_method_outputs_are_pinned(method, tmp_path):
    require_path("raw data/PROMISE12")
    got = fingerprint(tmp_path, method)
    golden = json.load(open(GOLDEN, encoding="utf-8")) if os.path.exists(GOLDEN) else {}
    if os.environ.get("UPDATE_GOLDEN") == "1":
        golden[method] = got
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w", encoding="utf-8") as f:
            json.dump(golden, f, indent=1, sort_keys=True)
        pytest.skip("golden value written")
    assert method in golden, "no golden value; run once with UPDATE_GOLDEN=1"
    assert got == golden[method]
