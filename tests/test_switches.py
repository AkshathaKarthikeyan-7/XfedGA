"""
1-round smoke tests for every method / scenario switch added in Session 3a, each with a check that
the switch actually did what it claims. Slow (GPU, real prostate data, tiny config).
"""
import json
import os

import numpy as np
import pytest
import torch

from tests.conftest import require_path

TINY = ["train.rounds=1", "train.local_epochs=1", "ga.population=2", "ga.generations=1",
        "fitness.subsample=8", "saliency.subsample=4", "evaluation.explain_samples=8"]


def _run(tmp_path, method, extra=(), seed=5):
    require_path("raw data/PROMISE12")
    from experiments.run_federated import run_experiment
    return run_experiment("prostate", method, seed, TINY + list(extra), out_root=str(tmp_path), evaluate_test=False)


def _row(res):
    return res["rounds"][0]


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("method", ["abl_B", "abl_C", "abl_D", "abl_F", "random_search", "fedprox"])
def test_method_switch_runs(method, tmp_path):
    res = _run(tmp_path, method)
    row = _row(res)
    assert 0.0 <= row["val_dsc"] <= 1.0
    q, alpha = json.loads(row["q_i"]), json.loads(row["alpha_i"])
    assert abs(sum(alpha) - 1.0) < 1e-6
    if method in ("abl_B", "abl_F"):
        assert len(q) == 2 and all(0.0 <= x <= 1.0 for x in q)
    else:
        assert q == []
    if method == "abl_B" or method == "fedprox":
        assert row["fitness_evals"] == 0
    if method == "random_search":
        from fedga.ga_optimizer import ga_evaluation_count
        assert row["fitness_evals"] == ga_evaluation_count(2, 1, True)


@pytest.mark.data
@pytest.mark.slow
def test_random_q_is_not_gradcam(tmp_path):
    f = json.loads(_row(_run(tmp_path / "f", "abl_F"))["q_i"])
    e = json.loads(_row(_run(tmp_path / "e", "xfedga"))["q_i"])
    assert f == np.random.RandomState(5 * 1000 + 1).uniform(0, 1, 2).round(6).tolist()
    assert f != e


@pytest.mark.data
@pytest.mark.slow
def test_fedprox_changes_the_local_update(tmp_path):
    a = _run(tmp_path / "a", "fedavg")["final_state"]
    b = _run(tmp_path / "b", "fedprox", ["method_options.fedprox_mu=1.0"])["final_state"]
    assert not all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.data
@pytest.mark.slow
def test_centralized_pools_clients(tmp_path):
    res = _run(tmp_path, "centralized")
    assert res["meta"]["n_i"] == [379 + 399] and json.loads(_row(res)["alpha_i"]) == [1.0]


@pytest.mark.data
@pytest.mark.slow
def test_local_only_trains_each_client_separately(tmp_path):
    res = _run(tmp_path, "local_only")
    assert set(res["meta"]["per_client"]) == {"PROMISE12_client1", "PROMISE12_client2"}
    assert os.path.exists(os.path.join(res["out"], "round_log_client1.csv"))
    assert os.path.exists(os.path.join(res["out"], "round_log_client2.csv"))


@pytest.mark.data
@pytest.mark.slow
def test_acquisition_partition(tmp_path):
    res = _run(tmp_path, "xfedga", ["scenario.partition=acquisition"])
    with open(os.path.join(res["out"], "split_log.json"), encoding="utf-8") as f:
        log = json.load(f)
    assert log["partition"] == "acquisition"
    assert [v["n_cases"] for v in log["clients"].values()] == [22, 28]


@pytest.mark.data
@pytest.mark.slow
def test_noisy_qi_client_reports_one(tmp_path):
    q = json.loads(_row(_run(tmp_path, "xfedga", ["scenario.noisy_qi_client=0"]))["q_i"])
    assert q[0] == 1.0 and q[1] < 1.0


@pytest.mark.data
@pytest.mark.slow
def test_dp_perturbs_updates(tmp_path):
    a = _run(tmp_path / "a", "fedavg")["final_state"]
    b = _run(tmp_path / "b", "fedavg", ["scenario.dp.enabled=true", "scenario.dp.clip_norm=1.0",
                                        "scenario.dp.noise_multiplier=0.1"])["final_state"]
    assert not all(torch.equal(a[k], b[k]) for k in a)


def test_privatize_update_clips_and_adds_noise():
    from experiments.config import torch_generator
    from fedga.client import privatize_update
    g = {"w": torch.zeros(1000)}
    s = {"w": torch.full((1000,), 0.1)}            # update norm = 0.1 * sqrt(1000) ~ 3.16
    clipped = privatize_update(s, g, clip_norm=1.0, noise_multiplier=0.0, generator=torch_generator(0))
    assert torch.isclose(clipped["w"].norm(), torch.tensor(1.0), atol=1e-4)
    noisy = privatize_update(s, g, clip_norm=1.0, noise_multiplier=0.5, generator=torch_generator(0))
    assert abs(float((noisy["w"] - clipped["w"]).std()) - 0.5) < 0.05


@pytest.mark.data
def test_corrupted_label_client_shuffles_masks():
    require_path("raw data/PROMISE12")
    from experiments.config import load_config
    from experiments.tasks import build_task
    clean = build_task("prostate", load_config("prostate"))
    bad = build_task("prostate", load_config("prostate", ["scenario.corrupted_label_client=1"]))
    assert bad.split_log["corrupted_label_client"] == "PROMISE12_client2"
    xs, ys = clean.clients[1].train[3], bad.clients[1].train[3]
    assert torch.equal(xs[0], ys[0]) and not torch.equal(xs[1], ys[1])
    assert torch.equal(clean.clients[0].train[3][1], bad.clients[0].train[3][1])   # other client untouched
