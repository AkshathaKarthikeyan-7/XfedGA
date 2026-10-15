"""Aggregation weights and the GA elitism guarantee (fedga/), on toy state dicts."""
import numpy as np
import pytest
import torch

from fedga.fedavg import explanation_weighted_aggregate, fedavg_aggregate
from fedga.ga_optimizer import run_ga


def _one_hot_states(k):
    """Client i holds 1.0 at position i: the aggregate of a key equals the weight vector."""
    return [{"w": torch.eye(k)[i].clone()} for i in range(k)]


@pytest.mark.parametrize("sizes", [[10, 30], [5, 5, 90], [1, 1]])
def test_fedavg_weights_sum_to_one_and_follow_sizes(sizes):
    agg = fedavg_aggregate(_one_hot_states(len(sizes)), sizes)["w"]
    assert torch.isclose(agg.sum(), torch.tensor(1.0))
    assert torch.allclose(agg, torch.tensor(sizes, dtype=torch.float32) / sum(sizes))


@pytest.mark.parametrize("beta", [0.0, 0.2, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("q", [[0.1, 0.6], [0.0, 0.9], [0.5, 0.5], [1e-6, 1e-6]])
def test_weighted_aggregation_weights_sum_to_one(beta, q):
    sizes = [40, 120]
    agg = explanation_weighted_aggregate(_one_hot_states(2), sizes, q, beta=beta)["w"]
    assert torch.isclose(agg.sum(), torch.tensor(1.0), atol=1e-6)
    assert (agg > 0).all()  # eps floor: no client is zeroed out
    raw = np.array([n / sum(sizes) * max(s, 1e-3) ** beta for n, s in zip(sizes, q)])
    assert torch.allclose(agg, torch.tensor(raw / raw.sum(), dtype=torch.float32), atol=1e-6)


def test_beta_zero_equals_fedavg():
    states, sizes = _one_hot_states(3), [3, 7, 10]
    a = explanation_weighted_aggregate(states, sizes, [0.05, 0.4, 0.9], beta=0.0)["w"]
    b = fedavg_aggregate(states, sizes)["w"]
    assert torch.allclose(a, b)


def _toy_problem(seed, target_offset):
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = {"a": torch.randn(50), "b": torch.randn(4, 4)}
    target = {k: v + target_offset * torch.randn_like(v) for k, v in base.items()}

    def fitness(sd):
        return -float(sum(((sd[k] - target[k]) ** 2).sum() for k in sd))
    return base, fitness


@pytest.mark.parametrize("mode", ["base", "xfedga"])
@pytest.mark.parametrize("seed", range(5))
def test_elitism_winner_never_worse_than_w_agg(mode, seed):
    base, fitness = _toy_problem(seed, target_offset=0.01)
    best = run_ga(base, fitness, mode=mode, population_size=6, generations=5, sigma0=0.01)
    assert fitness(best) >= fitness(base)


@pytest.mark.parametrize("mode", ["base", "xfedga"])
def test_w_agg_is_kept_when_it_is_the_optimum(mode):
    # target == w_agg: every perturbed candidate is strictly worse, so the GA must return w_agg itself
    base, fitness = _toy_problem(0, target_offset=0.0)
    best = run_ga(base, fitness, mode=mode, population_size=6, generations=5, sigma0=0.01)
    assert all(torch.equal(best[k], base[k]) for k in base)
