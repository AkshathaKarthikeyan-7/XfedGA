"""
XFedGA+ (declared post-v1 variant, plan.md): mixing-vector GA and server-verified client score.
  - every candidate mixing vector lies on the simplex; candidate 0 is alpha, i.e. exactly w_agg;
  - the fitness budget equals the weight-space GA's; elitism keeps fitness >= fitness(w_agg);
  - the server score uses only the fitness subsample (ServerEval never touches the test set);
  - 1-round smoke runs of both variants.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from fedga.fedavg import aggregate, aggregation_weights
from fedga.ga_optimizer import ga_evaluation_count, run_mix_ga
from tests.conftest import require_path


def _toy_states():
    g = torch.Generator().manual_seed(0)
    return [{"w": torch.randn(5, generator=g), "b": torch.randn(3, generator=g)} for _ in range(3)]


def test_mix_ga_simplex_wagg_first_budget_and_elitism():
    np.random.seed(0)
    states = _toy_states()
    alpha = [0.5, 0.3, 0.2]
    target = aggregate(states, [0.2, 0.2, 0.6])
    seen = []

    def fit(sd):
        seen.append(sd)
        return -float(sum(((sd[k] - target[k]) ** 2).sum() for k in sd))

    best, mix, hist, mixes = run_mix_ga(states, alpha, fit, population_size=6, generations=5, pc=0.5, kappa=20.0,
                                        tau=10.0, anneal=True, cache_elite_fitness=True)
    assert len(mixes) == len(seen) == ga_evaluation_count(6, 5, True) == sum(h["evaluations"] for h in hist)
    for m in mixes + [mix]:
        assert min(m) >= 0 and abs(sum(m) - 1) < 1e-9
    w_agg = aggregate(states, alpha)
    assert mixes[0] == alpha and all(torch.equal(seen[0][k], w_agg[k]) for k in w_agg)
    assert fit(best) >= fit(w_agg)
    assert all(torch.allclose(best[k], aggregate(states, mix)[k]) for k in best)


def test_mix_ga_is_deterministic_given_the_seed():
    states = _toy_states()
    runs = []
    for _ in range(2):
        np.random.seed(3)
        runs.append(run_mix_ga(states, [0.4, 0.4, 0.2], lambda sd: float(sd["w"].sum()), 6, 5, kappa=20.0,
                               cache_elite_fitness=True)[3])
    assert runs[0] == runs[1]


class _Images(Dataset):
    def __init__(self, n, size=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand(n, 1, size, size, generator=g)
        self.y = torch.zeros(n, size, size, dtype=torch.long)
        self.y[:, 20:40, 20:40] = 1

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _NoTest(dict):
    def __getitem__(self, k):
        if k == "test":
            raise AssertionError("the server must not read the test set")
        return super().__getitem__(k)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Grad-CAM targets need CUDA")
def test_server_score_uses_fitness_subset_only_and_kinds():
    from experiments.config import load_config
    from experiments.run_federated import METHODS, ServerEval, server_step
    from fedga.evaluation import FixedSet, server_client_scores
    from models.residual_unet import build_residual_unet
    dev = torch.device("cuda")
    torch.manual_seed(0)
    model = build_residual_unet().to(dev)
    states = []
    for k in range(2):
        torch.manual_seed(k + 1)
        states.append({n: v.detach().clone() for n, v in build_residual_unet().state_dict().items()})
    fixed = FixedSet(_Images(4), range(4), dev)
    a = server_client_scores(model, fixed, states, "dsc_ec", 64.0)
    b = server_client_scores(model, fixed, states, "dsc", 64.0)
    for x, y in zip(a, b):
        assert x["dsc"] == y["dsc"] and x["ec"] == y["ec"]
        assert x["s"] == pytest.approx(x["dsc"] * x["ec"]) and y["s"] == y["dsc"]
    with pytest.raises(ValueError):
        server_client_scores(model, fixed, states, "ec", 64.0)

    cfg = load_config("prostate", ["ga.population=2", "ga.generations=1", "fitness.hd95_normalizer=64",
                                   "image_size=[64,64]"])
    data = SimpleNamespace(server=_NoTest(fitness=SimpleNamespace(dataset=_Images(4, seed=1)),
                                          validation=SimpleNamespace(dataset=_Images(4, seed=2))),
                           fitness_subsample=[0, 1, 2, 3], val_explain_subsample=[0, 1])
    spec = METHODS["xfedga_plus"]
    srv = ServerEval(cfg, spec, data, dev)
    np.random.seed(0)
    row, _ = server_step(cfg, spec, 1, 1, model, states, [10, 30], [0.9, 0.1], srv, dev)
    s = json.loads(row["s_i"])
    assert json.loads(row["alpha_i"]) == pytest.approx(aggregation_weights([10, 30], s, cfg["aggregation"]["beta"],
                                                                           cfg["aggregation"]["eps"]), abs=1e-4)  # s_i logged to 6 d.p.
    assert json.loads(row["q_i"]) == [0.9, 0.1]                          # logged, not used
    m = json.loads(row["mix_i"])
    assert min(m) >= 0 and abs(sum(m) - 1) < 1e-5 and row["fitness_evals"] == ga_evaluation_count(2, 1, True)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.parametrize("method", ["xfedga_plus", "xfedga_plus_dsc"])
def test_xfedga_plus_smoke(method, tmp_path):
    require_path("raw data/PROMISE12")
    from tests.test_switches import _run
    row = _run(tmp_path, method, ["evaluation.log_client_val=true"])["rounds"][0]
    assert all(0.0 <= x <= 1.0 for x in json.loads(row["client_val_dsc_i"]))
    assert 0.0 <= row["val_ec_mass"] <= 1.0                      # both EC definitions logged
    s, alpha, mix = json.loads(row["s_i"]), json.loads(row["alpha_i"]), json.loads(row["mix_i"])
    dsc, ec = json.loads(row["client_fit_dsc_i"]), json.loads(row["client_fit_ec_i"])
    assert len(s) == len(alpha) == len(mix) == 2 and len(json.loads(row["q_i"])) == 2
    want = [d * e for d, e in zip(dsc, ec)] if method == "xfedga_plus" else dsc
    assert s == pytest.approx(want, abs=2e-6)
    assert abs(sum(mix) - 1) < 1e-5 and 0.0 <= row["val_dsc"] <= 1.0
