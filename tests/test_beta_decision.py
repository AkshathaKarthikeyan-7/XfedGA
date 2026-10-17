"""The pre-registered beta selection rule (analysis/beta_decision.py), on synthetic screen results."""
import sys

import pytest

sys.path.insert(0, "analysis")
from beta_decision import decide  # noqa: E402


def runs(spec):
    """spec: {beta: ([score_seed1, score_seed2], q_var)} -> synthetic run records for task 'prostate'."""
    out = []
    for beta, (scores, qv) in spec.items():
        for seed, sc in enumerate(scores, 1):
            out.append({"task": "prostate", "seed": seed, "beta": beta, "score": sc, "q_var": qv, "alpha_var": 0.0})
    return out


# every beta has seed scores m +/- 0.01 -> seed s.d. = 0.01414, pooled s.d. = 0.01414
def pair(m):
    return [m - 0.01, m + 0.01]


def test_default_kept_when_nothing_beats_it_by_more_than_pooled_sd():
    d = decide(runs({0.0: (pair(0.60), 1), 0.2: (pair(0.61), 1), 0.5: (pair(0.60), 1), 1.0: (pair(0.585), 1)}), "prostate")
    assert d["pooled_sd"] == pytest.approx(0.0141421, abs=1e-6)
    assert d["best_beta"] == 0.5 and d["betas_beating_default"] == []


def test_switch_when_one_beta_clears_the_margin():
    d = decide(runs({0.0: (pair(0.60), 1), 0.2: (pair(0.63), 1), 0.5: (pair(0.60), 1), 1.0: (pair(0.59), 1)}), "prostate")
    assert d["best_beta"] == 0.2 and d["betas_beating_default"] == [0.2]


def test_several_within_one_sd_tie_broken_by_lower_q_variance():
    d = decide(runs({0.0: (pair(0.63), 5.0), 0.2: (pair(0.64), 9.0), 0.5: (pair(0.60), 1), 1.0: (pair(0.59), 1)}), "prostate")
    assert set(d["betas_beating_default"]) == {0.0, 0.2}
    assert d["best_beta"] == 0.0          # 0.2 scores higher but is within one s.d.; 0.0 has the lower q_i variance


def test_several_far_apart_highest_wins():
    d = decide(runs({0.0: (pair(0.62), 5.0), 0.2: (pair(0.70), 9.0), 0.5: (pair(0.60), 1), 1.0: (pair(0.59), 1)}), "prostate")
    assert d["best_beta"] == 0.2 and d["tied_after_beating"] == [0.2]


def test_margin_must_exceed_not_equal_the_pooled_sd():
    sd = 0.0141421356
    d = decide(runs({0.0: (pair(0.60 + sd), 1), 0.2: (pair(0.60), 1), 0.5: (pair(0.60), 1), 1.0: (pair(0.60), 1)}), "prostate")
    assert d["best_beta"] == 0.5
