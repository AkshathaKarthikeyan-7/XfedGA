"""Acceptance check (pre-registered 18 Oct 2026): fitness halves A / B, the accept / reject rule, method restriction."""
import json

import numpy as np
import pytest
import torch

from data.splits import fitness_halves


def test_halves_are_case_level_disjoint_complete_and_deterministic():
    rng = np.random.RandomState(0)
    groups = [f"c{k}" for k in range(8) for _ in range(5)]                 # 8 cases x 5 slices (prostate-like)
    sizes = list(rng.rand(40))
    a, b = fitness_halves(groups, sizes, 42)
    assert sorted(a + b) == list(range(40)) and not set(a) & set(b)
    assert not {groups[i] for i in a} & {groups[i] for i in b}             # no case in both halves
    assert len({groups[i] for i in a}) == len({groups[i] for i in b}) == 4
    assert (a, b) == fitness_halves(groups, sizes, 42)
    one = [f"p{k}" for k in range(40)]                                       # breast-like: one image per patient
    a1, b1 = fitness_halves(one, sizes, 42)
    assert len(a1) == len(b1) == 20


def test_halves_are_balanced_by_size():
    sizes = list(np.linspace(0.01, 0.4, 40))
    groups = [f"p{k}" for k in range(40)]
    a, b = fitness_halves(groups, sizes, 42)
    gap = abs(np.mean([sizes[i] for i in a]) - np.mean([sizes[i] for i in b]))
    assert gap < 0.02                                                        # blocks of 2 similar sizes


def test_acceptance_decision():
    from experiments.run_federated import acceptance_decision
    win, agg = {"w": torch.ones(1)}, {"w": torch.zeros(1)}
    state, ok, fw, fa = acceptance_decision(win, agg, lambda s: float(s["w"]))           # winner better on B
    assert ok and state is win and (fw, fa) == (1.0, 0.0)
    state, ok, _, _ = acceptance_decision(win, agg, lambda s: -float(s["w"]))            # winner worse on B
    assert not ok and state is agg
    state, ok, _, _ = acceptance_decision(win, agg, lambda s: 0.5)                        # tie -> accepted
    assert ok and state is win


def test_acceptance_check_is_restricted_to_xfedga_variants(tmp_path):
    from experiments.run_federated import run_experiment
    with pytest.raises(ValueError):
        run_experiment("breast", "fedga", 1, ["ga.acceptance_check=true"], out_root=str(tmp_path), evaluate_test=False)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("method", ["xfedga", "xfedga_plus"])
def test_acceptance_check_smoke(method, tmp_path):
    from tests.test_switches import _run
    row = _run(tmp_path, method, ["ga.acceptance_check=true"])["rounds"][0]
    assert row["accepted"] in (0, 1) and (row["fit_B_winner"] >= row["fit_B_wagg"]) == bool(row["accepted"])
    if method == "xfedga_plus" and not row["accepted"]:
        assert json.loads(row["mix_i"]) == pytest.approx(json.loads(row["alpha_i"]), abs=1e-5)
