"""
Statistics for the results tables (spec section 6 and 9B). Pure functions, no I/O.

Protocol:
  - per-image test rows are averaged to one value per case (patient for BUS-BRA, case for Decathlon);
  - for each method, per-case values are averaged over seeds (seeds quantify training variance and
    are reported separately);
  - two methods are compared on the cases they share: d_c = value_A(c) - value_B(c);
  - 95% CI of mean(d) by cluster bootstrap over cases (resample cases with replacement);
  - effect size: Cohen's d_z = mean(d) / sd(d) for the paired case differences;
  - outcome: WIN if the CI excludes 0 in the favourable direction, LOSE if it excludes 0 in the
    unfavourable direction, otherwise TIE.
"""
import math
from collections import defaultdict

import numpy as np

# direction of each metric: +1 = higher is better, -1 = lower is better
DIRECTION = {"dsc": +1, "hd95_norm": -1, "hd95_px": -1, "ec": +1, "pg": +1,
             "pg_gradcam": +1, "pg_gradcampp": +1, "iou_gradcampp": +1, "deletion_auc": -1, "insertion_auc": +1}


def case_means(rows, metric):
    """rows: dicts with group_id and metric -> {case: mean over that case's images (NaN ignored)}."""
    acc = defaultdict(list)
    for r in rows:
        v = float(r[metric])
        if not math.isnan(v):
            acc[r["group_id"]].append(v)
    return {c: float(np.mean(v)) for c, v in acc.items() if v}


def seed_averaged(per_seed):
    """per_seed: list of {case: value} (one per seed) -> {case: mean over the seeds that have the case}."""
    acc = defaultdict(list)
    for d in per_seed:
        for c, v in d.items():
            acc[c].append(v)
    return {c: float(np.mean(v)) for c, v in acc.items()}


def paired_differences(a, b):
    cases = sorted(set(a) & set(b))
    return cases, np.array([a[c] - b[c] for c in cases])


def bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=0):
    """Percentile CI of mean(d) by resampling clusters (cases) with replacement."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def effect_size_dz(d):
    d = np.asarray(d, dtype=float)
    sd = d.std(ddof=1) if d.size > 1 else float("nan")
    return float(d.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else float("nan")


def outcome(ci_low, ci_high, direction):
    """WIN / TIE / LOSE for the difference (method - comparator) of a metric with the given direction."""
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "n/a"
    if ci_low > 0:
        return "WIN" if direction > 0 else "LOSE"
    if ci_high < 0:
        return "LOSE" if direction > 0 else "WIN"
    return "TIE"


def compare(method_runs, comparator_runs, metric, n_boot=10000, seed=0):
    """
    method_runs / comparator_runs: lists of per-image row lists (one list per seed).
    Returns the paired case-level comparison of `metric` (method - comparator).
    """
    a = seed_averaged([case_means(r, metric) for r in method_runs])
    b = seed_averaged([case_means(r, metric) for r in comparator_runs])
    cases, d = paired_differences(a, b)
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return {"metric": metric, "n_cases": len(cases), "mean_diff": float(d.mean()) if d.size else float("nan"),
            "ci_low": lo, "ci_high": hi, "effect_dz": effect_size_dz(d),
            "outcome": outcome(lo, hi, DIRECTION[metric]),
            "n_seeds": (len(method_runs), len(comparator_runs))}


def mean_sd_over_seeds(per_seed_values):
    v = np.asarray(per_seed_values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float("nan")
