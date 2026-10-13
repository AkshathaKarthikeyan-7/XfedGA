"""
GA-based server-side optimization.

Contains two modes, selected via `mode`:
  - "base"      : reproduces the original FedGA (Ahmed et al.) exactly --
                   single-objective DSC fitness, fixed-sigma Gaussian mutation.
  - "xfedga"    : your contribution -- multi-objective fitness
                   (DSC + HD95 + explanation consistency) and annealed mutation.
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


def run_ga(base_state_dict, evaluate_fn, mode="xfedga",
           population_size=10, generations=20,
           pc=0.5, pm=0.1, sigma0=0.01, tau=10.0):
    population = initial_population(base_state_dict, population_size, sigma0)

    for g in range(generations):
        sigma = sigma0 if mode == "base" else annealed_sigma(sigma0, g, tau)

        fitnesses = [evaluate_fn(ind) for ind in population]
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        population = [ind for ind, _ in ranked]

        parents = population[: population_size // 2]

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

    final_fitnesses = [evaluate_fn(ind) for ind in population]
    best_idx = int(np.argmax(final_fitnesses))
    return population[best_idx]
