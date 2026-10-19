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
| _v2screen | breast | FedGA (baseline) | [1, 2] | [0.0, 0.0] | 0.13 | 0.0066 | — | — | [9, 6] | 0.690 → 0.642 (-0.049) |
| _v2screen | breast | XFedGA | [1, 2] | [0.03078, 0.03078] | 0.20 | 0.0038 | — | — | [5, 13] | 0.687 → 0.634 (-0.052) |
| _v2screen | prostate | FedGA (baseline) | [1, 2] | [0.0, 0.0] | 0.40 | 0.0013 | — | — | [15, 15] | 0.649 → 0.649 (-0.000) |
| _v2screen | prostate | XFedGA | [1, 2] | [0.03452, 0.03452] | 0.47 | 0.0005 | — | — | [15, 14] | 0.651 → 0.638 (-0.013) |
| _xplusscreen | breast | XFedGA+ | [1, 2] | [0.01559, 0.01559] | 0.07 | 0.0144 | 0.83 | 0.39 | [2, 3] | 0.694 → 0.609 (-0.085) |
| _xplusscreen | breast | XFedGA+ (DSC-only) | [1, 2] | [0.00754, 0.00754] | 0.03 | 0.0145 | 0.93 | 0.72 | [3, 3] | 0.681 → 0.615 (-0.066) |
| _msv1screen | prostate_ms | FedGA (baseline) | [1, 2] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0.20 | 0.0084 | — | — | [5, 8] | 0.720 → 0.604 (-0.117) |
| _msv1screen | prostate_ms | XFedGA | [1, 2] | [0.01464, 0.00892, 0.0397, 0.04275, 0.00858] | 0.20 | 0.0096 | — | — | [3, 3] | 0.690 → 0.608 (-0.082) |
| _msv1screen | prostate_ms | XFedGA+ | [1, 2] | [0.02485, 0.01288, 0.06724, 0.03848, 0.02924] | 0.00 | 0.0419 | — | — | [4, 3] | 0.813 → 0.771 (-0.042) |
| _msv1screen | prostate_ms | XFedGA+ (DSC-only) | [1, 2] | [0.01238, 0.00988, 0.02902, 0.01975, 0.01743] | 0.00 | 0.0631 | — | — | [3, 7] | 0.790 → 0.739 (-0.051) |
| _msv2screen | prostate_ms | FedGA (baseline) | [1, 2] | [0.0, 0.0, 0.0, 0.0, 0.0] | 0.17 | 0.0090 | — | — | [14, 15] | 0.842 → 0.840 (-0.002) |
| _msv2screen | prostate_ms | XFedGA | [1, 2] | [0.0296, 0.01774, 0.05081, 0.02295, 0.01521] | 0.20 | 0.0073 | — | — | [14, 13] | 0.836 → 0.832 (-0.004) |
| _msv2screen | prostate_ms | XFedGA+ | [1, 2] | [0.01584, 0.00596, 0.01677, 0.02457, 0.00889] | 0.00 | 0.0214 | — | — | [13, 15] | 0.853 → 0.849 (-0.004) |
| _msv2screen | prostate_ms | XFedGA+ (DSC-only) | [1, 2] | [0.00772, 0.00268, 0.01304, 0.01283, 0.0067] | 0.00 | 0.0247 | — | — | [12, 15] | 0.853 → 0.849 (-0.004) |

RC6 (heterogeneity / corrupted client) is addressed by the multi-site prostate and dose-response experiments.
