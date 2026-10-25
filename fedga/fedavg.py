"""Standard FedAvg aggregation, plus the explanation-weighted variant."""
import copy


def fedavg_aggregate(client_state_dicts, client_sizes):
    """FedAvg: weighted average of the client state dicts with weights n_i / N."""
    return _weighted_average(client_state_dicts, aggregation_weights(client_sizes))


def aggregation_weights(client_sizes, client_saliency_scores=None, beta=0.5, eps=1e-3):
    """
    Normalised aggregation weights alpha_i.
      FedAvg (scores None):  alpha_i = n_i / N
      XFedGA:                alpha_i ∝ (n_i / N) * max(q_i, eps) ** beta
    eps floors q_i so a client with momentarily poor saliency is down-weighted, never zeroed.
    beta = 0 reduces to FedAvg. beta is a CLI/YAML setting (aggregation.beta); its value is
    chosen on the validation set (plan.md section 0), not fixed by argument here.
    """
    total = sum(client_sizes)
    if client_saliency_scores is None:
        raw = [n / total for n in client_sizes]
    else:
        raw = [(n / total) * (max(s, eps) ** beta) for n, s in zip(client_sizes, client_saliency_scores)]
    norm = sum(raw)
    return [w / norm for w in raw]


def aggregate(client_state_dicts, weights):
    """w_agg = sum_i alpha_i * w_i (weights from aggregation_weights)."""
    return _weighted_average(client_state_dicts, weights)


def explanation_weighted_aggregate(client_state_dicts, client_sizes, client_saliency_scores,
                                    beta=0.5, eps=1e-3):
    """XFedGA aggregation: weight by data size AND client saliency score q_i (see aggregation_weights)."""
    return _weighted_average(client_state_dicts,
                             aggregation_weights(client_sizes, client_saliency_scores, beta, eps))


def _weighted_average(state_dicts, weights):
    avg_state = copy.deepcopy(state_dicts[0])
    for key in avg_state.keys():
        avg_state[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))
    return avg_state