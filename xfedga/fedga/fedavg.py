"""Standard FedAvg aggregation, plus the explanation-weighted variant."""
import copy
import torch


def fedavg_aggregate(client_state_dicts, client_sizes):
    total = sum(client_sizes)
    weights = [n / total for n in client_sizes]
    return _weighted_average(client_state_dicts, weights)


def explanation_weighted_aggregate(client_state_dicts, client_sizes, client_saliency_scores,
                                    beta=0.5, eps=1e-3):
    """
    XFedGA aggregation: weight by data size AND explanation quality.
    client_saliency_scores[i] in [0,1] -- how well client i's own Grad-CAM
    aligns with its own ground truth. eps floors the weight so a client with
    momentarily poor saliency isn't zeroed out entirely (just downweighted).
    beta=0.5 (dampened from 1.0): per-client saliency scores are estimated
    from only ~20 sampled images and swing noisily round to round (observed
    range ~0.04-0.20 for the same client across rounds); beta=1.0 let that
    noise directly inject variance into the aggregated model. beta=0.5
    still lets explanation quality influence aggregation, just less sharply.
    """
    total_n = sum(client_sizes)
    raw_weights = [
        (n / total_n) * (max(s, eps) ** beta)
        for n, s in zip(client_sizes, client_saliency_scores)
    ]
    norm = sum(raw_weights)
    weights = [w / norm for w in raw_weights]
    return _weighted_average(client_state_dicts, weights)


def _weighted_average(state_dicts, weights):
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        avg_state[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    return avg_state