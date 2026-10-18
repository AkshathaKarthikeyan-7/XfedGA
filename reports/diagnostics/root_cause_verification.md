# Root-cause verification

Source: `analysis/root_cause.py` → `results/diagnostics/root_cause.json`. Validation data and training logs only; no test data. A dash means the quantity was not logged for that run group.

| Group | Task | Method | Seeds | RC1 mean abs(alpha − n/N) per client | RC2 rounds winner < w_agg (val) | RC3 mean winner − w_agg (val DSC) | RC4 higher score ⇒ higher val DSC | RC4 pooled Spearman | RC5 peak rounds | RC5 peak → last |
|---|---|---|---|---|---|---|---|---|---|---|
| v1_headline | breast | FedAvg | [1, 2, 3, 4, 5] | — | — | — | — | — | [2, 2, 2, 3, 3] | 0.666 → 0.590 (-0.076) |
| v1_headline | breast | FedGA (baseline) | [1, 2, 3, 4, 5] | [0.0, 0.0] | — | — | — | — | [2, 4, 2, 3, 2] | 0.675 → 0.597 (-0.078) |
| v1_headline | breast | XFedGA | [1, 2, 3, 4, 5] | [0.0185, 0.0185] | — | — | — | — | [2, 2, 2, 2, 2] | 0.679 → 0.594 (-0.085) |
| v1_headline | prostate | FedAvg | [1, 2, 3, 4, 5] | — | — | — | — | — | [8, 13, 4, 6, 1] | 0.440 → 0.354 (-0.086) |
| v1_headline | prostate | FedGA (baseline) | [1, 2, 3, 4, 5] | [0.0, 0.0] | — | — | — | — | [5, 5, 5, 1, 1] | 0.432 → 0.351 (-0.081) |
| v1_headline | prostate | XFedGA | [1, 2, 3, 4, 5] | [0.01224, 0.01224] | — | — | — | — | [5, 5, 7, 2, 3] | 0.461 → 0.345 (-0.115) |
| _v2screen | breast | FedGA (baseline) | [1] | [0.0, 0.0] | 0.13 | 0.0054 | — | — | [9] | 0.679 → 0.601 (-0.078) |
| _v2screen | breast | XFedGA | [1] | [0.02798, 0.02798] | 0.20 | 0.0047 | — | — | [5] | 0.683 → 0.630 (-0.053) |
| _v2screen | prostate | FedGA (baseline) | [1, 2] | [0.0, 0.0] | 0.40 | 0.0013 | — | — | [15, 15] | 0.649 → 0.649 (-0.000) |
| _v2screen | prostate | XFedGA | [1] | [0.0368, 0.0368] | 0.53 | 0.0013 | — | — | [15] | 0.638 → 0.638 (-0.000) |

RC6 (heterogeneity / corrupted client) is addressed by the multi-site prostate and dose-response experiments.
