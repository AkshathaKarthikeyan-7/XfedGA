"""
GA-based server-side optimization.

Two mutation schedules, selected via `mode`:
  - "base"   : FedGA (baseline) -- the GA operators of Algorithm 1 of the base FedGA paper
               (elitist top-half selection, uniform crossover, sparse Gaussian mutation with
               fixed sigma). Population size, generations and fitness set are our settings,
               which differ from the paper (reports/audit/fedga_paper_comparison.md).
  - "xfedga" : same operators, mutation sigma annealed per generation: sigma0 * exp(-g / tau).
The fitness function is passed in (evaluate_fn); see fedga/evaluation.py.
"""
import copy
import math
import torch
import numpy as np
from monai.metrics import HausdorffDistanceMetric

from xai.explainability import explanation_consistency  # single source of truth

_hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")


def add_gaussian_noise(state_dict, sigma):
    noisy = copy.deepcopy(state_dict)
    for key in noisy.keys():
        if noisy[key].dtype.is_floating_point:
            noisy[key] = noisy[key] + torch.randn_like(noisy[key]) * sigma
    return noisy


def initial_population(base_state_dict, population_size=10, sigma=0.01):
    population = [copy.deepcopy(base_state_dict)]
    for _ in range(population_size - 1):
        population.append(add_gaussian_noise(base_state_dict, sigma))
    return population


def uniform_crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    for key in child.keys():
        if child[key].dtype.is_floating_point:
            mask = torch.rand_like(child[key]) < 0.5
            child[key] = torch.where(mask, parent1[key], parent2[key])
    return child


def mutate(state_dict, sigma, alpha):
    mutated = copy.deepcopy(state_dict)
    for key in mutated.keys():
        if mutated[key].dtype.is_floating_point:
            mask = (torch.rand_like(mutated[key]) < alpha).float()
            noise = torch.randn_like(mutated[key]) * sigma
            mutated[key] = mutated[key] + mask * noise
    return mutated


def dice_score(pred, target, eps=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def hd95(pred, target, max_dist=384.0):
    """
    95th-percentile Hausdorff Distance, via MONAI's HausdorffDistanceMetric.
    pred, target: (H,W) binary tensors (single image, single class).
    Returns a value normalized by max_dist (image diagonal-ish scale) so it's
    roughly comparable in magnitude to DSC/EC for the multi-objective fitness
    combination -- raw HD95 is in pixels and would otherwise dominate the sum.
    If pred and target are both empty (no foreground), returns 0 (perfect).
    """
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return 1.0  # worst case, one side has nothing to compare
    p = pred.unsqueeze(0).unsqueeze(0).float()   # (1,1,H,W)
    t = target.unsqueeze(0).unsqueeze(0).float()
    _hd95_metric.reset()
    _hd95_metric(y_pred=p, y=t)
    raw = _hd95_metric.aggregate().item()
    return min(raw / max_dist, 1.0)  # clip/normalize to [0,1]


def multi_objective_fitness(dsc, hd95_norm, ec, weights=(0.5, 0.3, 0.2)):
    """F(M) = w1*DSC - w2*HD95_norm + w3*EC"""
    w1, w2, w3 = weights
    return w1 * dsc - w2 * hd95_norm + w3 * ec


def annealed_sigma(sigma0, generation, tau=10.0):
    return sigma0 * math.exp(-generation / tau)


def _evaluate(population, known, evaluate_fn):
    """Fitness of every individual; entries already in `known` (cached elites) are reused."""
    fits, n = [], 0
    for ind, f in zip(population, known):
        if f is None:
            f = evaluate_fn(ind)
            n += 1
        fits.append(f)
    return fits, n


def run_ga(base_state_dict, evaluate_fn, mode="xfedga",
           population_size=10, generations=20,
           pc=0.5, pm=0.1, sigma0=0.01, tau=10.0,
           cache_elite_fitness=False, return_history=False):
    """
    Elitist GA around w_agg = base_state_dict; returns the fittest final individual
    (and, with return_history, one {"best", "mean", "evaluations"} dict per evaluation pass).
    w_agg is individual 0 of the initial population and the top half always survives, so the
    winner's fitness is >= fitness(w_agg) whenever evaluate_fn is deterministic.
    cache_elite_fitness reuses the stored fitness of surviving elites instead of re-scoring
    them (P + G*P/2 instead of P*(G+1) evaluations); enable it only if evaluation is
    deterministic (decision Q7, tests/test_determinism.py).
    """
    population = initial_population(base_state_dict, population_size, sigma0)
    known = [None] * len(population)
    history = []

    for g in range(generations):
        sigma = sigma0 if mode == "base" else annealed_sigma(sigma0, g, tau)

        fitnesses, n_eval = _evaluate(population, known, evaluate_fn)
        history.append({"best": max(fitnesses), "mean": float(np.mean(fitnesses)), "evaluations": n_eval})
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        population = [ind for ind, _ in ranked]

        parents = population[: population_size // 2]
        parent_fitness = [f for _, f in ranked[: population_size // 2]]

        offspring = []
        while len(offspring) < population_size - len(parents):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            if np.random.rand() < pc:
                child = uniform_crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)
            child = mutate(child, sigma, pm)
            offspring.append(child)

        population = parents + offspring
        known = (parent_fitness if cache_elite_fitness else [None] * len(parents)) + [None] * len(offspring)

    final_fitnesses, n_eval = _evaluate(population, known, evaluate_fn)
    history.append({"best": max(final_fitnesses), "mean": float(np.mean(final_fitnesses)), "evaluations": n_eval})
    best = population[int(np.argmax(final_fitnesses))]
    return (best, history) if return_history else best


def ga_evaluation_count(population_size, generations, cache_elite_fitness):
    """Fitness evaluations one run_ga call performs (used to give random search an equal budget)."""
    if cache_elite_fitness:
        return population_size + generations * (population_size - population_size // 2)
    return population_size * (generations + 1)


def random_search(base_state_dict, evaluate_fn, n_evaluations, sigma0=0.01):
    """
    GA-free control (spec section 5): score w_agg and (n_evaluations - 1) independent Gaussian
    neighbours w_agg + N(0, sigma0^2), same noise model as the GA's initial population, and keep the
    best. Same fitness budget as the GA it is compared with; no crossover, mutation or selection.
    Returns (best, history) with one history entry, like run_ga(return_history=True).
    """
    best, best_fit = base_state_dict, evaluate_fn(base_state_dict)
    fits = [best_fit]
    for _ in range(n_evaluations - 1):
        cand = add_gaussian_noise(base_state_dict, sigma0)
        f = evaluate_fn(cand)
        fits.append(f)
        if f > best_fit:
            best, best_fit = cand, f
    return best, [{"best": best_fit, "mean": float(np.mean(fits)), "evaluations": len(fits)}]


def _dirichlet(conc):
    return np.random.dirichlet(np.maximum(np.asarray(conc, dtype=np.float64), 1e-3))


def run_mix_ga(client_states, alpha, evaluate_fn, population_size=10, generations=20, pc=0.5, kappa=20.0,
               tau=10.0, anneal=True, cache_elite_fitness=False):
    """
    XFedGA+ (declared post-v1 variant, plan.md): elitist GA over CLIENT MIXING VECTORS m on the simplex;
    a candidate model is sum_k m_k w_k. Individual 0 is m = alpha, i.e. exactly w_agg; the others are drawn
    from Dirichlet(kappa * alpha). Same selection, population, generations and elite caching as run_ga
    (P + G*P/2 evaluations with the cache), so the fitness budget equals XFedGA's.
      crossover (prob. pc): convex blend u*m1 + (1-u)*m2, u ~ U(0, 1)
      mutation (every offspring): m' ~ Dirichlet(kappa_g * m); kappa_g = kappa * exp(g / tau) if anneal
                                  (the analogue of XFedGA's sigma0 * exp(-g / tau): smaller steps later)
    Returns (best_state, best_mix, history, evaluated_mixes); history as run_ga(return_history=True).
    """
    from fedga.fedavg import aggregate

    alpha = np.asarray(alpha, dtype=np.float64)
    population = [alpha.copy()] + [_dirichlet(kappa * alpha) for _ in range(population_size - 1)]
    known = [None] * len(population)
    history, evaluated = [], []

    def score(mixes, known):
        fits, n = [], 0
        for m, f in zip(mixes, known):
            if f is None:
                f = evaluate_fn(aggregate(client_states, m.tolist()))
                evaluated.append(m.tolist())
                n += 1
            fits.append(f)
        return fits, n

    for g in range(generations):
        kg = kappa * math.exp(g / tau) if anneal else kappa
        fitnesses, n_eval = score(population, known)
        history.append({"best": max(fitnesses), "mean": float(np.mean(fitnesses)), "evaluations": n_eval})
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        parents = [m for m, _ in ranked[: population_size // 2]]
        parent_fitness = [f for _, f in ranked[: population_size // 2]]
        offspring = []
        while len(offspring) < population_size - len(parents):
            p1, p2 = parents[np.random.randint(len(parents))], parents[np.random.randint(len(parents))]
            if np.random.rand() < pc:
                u = np.random.rand()
                child = u * p1 + (1 - u) * p2
            else:
                child = p1.copy()
            offspring.append(_dirichlet(kg * child))
        population = parents + offspring
        known = (parent_fitness if cache_elite_fitness else [None] * len(parents)) + [None] * len(offspring)

    final, n_eval = score(population, known)
    history.append({"best": max(final), "mean": float(np.mean(final)), "evaluations": n_eval})
    best_mix = population[int(np.argmax(final))]
    return aggregate(client_states, best_mix.tolist()), best_mix.tolist(), history, evaluated
