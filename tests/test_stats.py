"""Bootstrap, pairing and WIN/TIE/LOSE logic of analysis/stats.py on synthetic data."""
import sys

import numpy as np
import pytest

sys.path.insert(0, "analysis")
from stats import (bootstrap_ci, case_means, compare, effect_size_dz, outcome, paired_differences,  # noqa: E402
                   seed_averaged)


def test_constant_difference_gives_degenerate_ci():
    lo, hi = bootstrap_ci(np.full(16, 0.03))
    assert lo == pytest.approx(0.03) and hi == pytest.approx(0.03)


def test_bootstrap_ci_coverage_is_close_to_95_percent():
    rng = np.random.default_rng(1)
    hits = 0
    for i in range(300):
        d = rng.normal(0.02, 0.07, size=16)          # prostate-like: 16 test cases, SD 0.07
        lo, hi = bootstrap_ci(d, n_boot=2000, seed=i)
        hits += lo <= 0.02 <= hi
    assert 0.88 <= hits / 300 <= 0.98                 # percentile bootstrap is slightly liberal at n = 16


def test_bootstrap_is_reproducible_with_seed():
    d = np.random.default_rng(0).normal(size=30)
    assert bootstrap_ci(d, seed=5) == bootstrap_ci(d, seed=5)


def test_case_means_average_images_within_case_and_skip_nan():
    rows = [{"group_id": "A", "dsc": 0.2}, {"group_id": "A", "dsc": 0.4}, {"group_id": "B", "dsc": float("nan")},
            {"group_id": "B", "dsc": 0.9}]
    assert case_means(rows, "dsc") == {"A": pytest.approx(0.3), "B": 0.9}


def test_seeds_are_averaged_within_case_before_pairing():
    per_seed = [{"A": 0.1, "B": 0.5}, {"A": 0.3, "B": 0.7}, {"A": 0.2}]
    assert seed_averaged(per_seed) == {"A": pytest.approx(0.2), "B": pytest.approx(0.6)}
    cases, d = paired_differences({"A": 0.5, "B": 0.6, "C": 0.1}, {"A": 0.4, "B": 0.6})
    assert cases == ["A", "B"] and np.allclose(d, [0.1, 0.0])


@pytest.mark.parametrize("lo,hi,direction,expected", [
    (0.01, 0.05, +1, "WIN"), (-0.05, -0.01, +1, "LOSE"), (-0.01, 0.02, +1, "TIE"),
    (-0.05, -0.01, -1, "WIN"), (0.01, 0.05, -1, "LOSE"), (-0.02, 0.01, -1, "TIE"),
    (0.0, 0.03, +1, "TIE"), (float("nan"), 0.1, +1, "n/a")])
def test_outcome_rule(lo, hi, direction, expected):
    assert outcome(lo, hi, direction) == expected


def test_effect_size_dz():
    assert effect_size_dz([0.1, 0.3]) == pytest.approx(0.2 / np.std([0.1, 0.3], ddof=1))
    assert np.isnan(effect_size_dz([0.1]))


def _runs(case_values, n_seeds, noise, rng):
    """per-image rows for n_seeds runs: two images per case around case_values[c]."""
    out = []
    for _ in range(n_seeds):
        rows = []
        for c, v in case_values.items():
            for _ in range(2):
                rows.append({"group_id": c, "dsc": v + rng.normal(0, noise), "hd95_norm": 0.3 - v / 10})
        out.append(rows)
    return out


def test_compare_detects_a_real_improvement_and_calls_ties():
    rng = np.random.default_rng(3)
    base = {f"c{i}": 0.5 + 0.1 * rng.normal() for i in range(40)}
    better = {c: v + 0.05 for c, v in base.items()}
    win = compare(_runs(better, 5, 0.01, rng), _runs(base, 5, 0.01, rng), "dsc")
    assert win["outcome"] == "WIN" and win["n_cases"] == 40 and win["mean_diff"] == pytest.approx(0.05, abs=0.01)
    tie = compare(_runs(base, 5, 0.05, rng), _runs(base, 5, 0.05, rng), "dsc")
    assert tie["outcome"] == "TIE"
    hd = compare(_runs(better, 3, 0.01, rng), _runs(base, 3, 0.01, rng), "hd95_norm")   # lower HD95 = better
    assert hd["mean_diff"] < 0 and hd["outcome"] == "WIN"


def test_t_interval_and_wilcoxon():
    from scipy import stats as st
    from stats import t_interval, wilcoxon_p
    d = np.array([0.01, 0.03, -0.02, 0.05, 0.04, 0.00, 0.02, 0.06])
    lo, hi = t_interval(d)
    half = st.t.ppf(0.975, 7) * d.std(ddof=1) / np.sqrt(8)
    assert lo == pytest.approx(d.mean() - half) and hi == pytest.approx(d.mean() + half)
    assert wilcoxon_p(np.zeros(5)) == 1.0
    assert 0 < wilcoxon_p(d) < 1


def test_compare_reports_sensitivity_and_flags_disagreement():
    rng = np.random.default_rng(7)
    base = {f"c{i}": 0.5 for i in range(16)}
    better = {c: 0.5 + 0.02 * (1 + rng.standard_t(2)) for c in base}   # heavy-tailed differences, n = 16
    r = compare([[{"group_id": c, "dsc": v} for c, v in better.items()]],
                [[{"group_id": c, "dsc": v} for c, v in base.items()]], "dsc")
    for k in ("t_ci_low", "t_ci_high", "t_outcome", "wilcoxon_p", "bootstrap_t_disagree"):
        assert k in r
    assert r["bootstrap_t_disagree"] == (r["outcome"] != r["t_outcome"])
