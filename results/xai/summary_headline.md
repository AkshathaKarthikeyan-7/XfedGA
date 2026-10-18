| Task | Method | Seeds | pg_gradcampp | rho_after_all_layers | pg_gradcam | iou_gradcampp | deletion_auc | insertion_auc |
|---|---|---|---|---|---|---|---|---|
| prostate | FedAvg | 5 | 0.002 ± 0.006 | -0.102 ± 0.046 | 0.592 ± 0.055 | 0.005 ± 0.006 | 0.273 ± 0.079 | 0.253 ± 0.040 |
| prostate | FedGA (baseline) | 5 | 0.000 ± 0.000 | -0.115 ± 0.062 | 0.605 ± 0.073 | 0.002 ± 0.002 | 0.236 ± 0.141 | 0.254 ± 0.121 |
| prostate | XFedGA | 5 | 0.001 ± 0.002 | -0.086 ± 0.076 | 0.611 ± 0.026 | 0.004 ± 0.003 | 0.272 ± 0.138 | 0.253 ± 0.123 |
| breast | FedAvg | 5 | 0.154 ± 0.342 | 0.084 ± 0.043 | 0.788 ± 0.036 | 0.026 ± 0.056 | 0.722 ± 0.097 | 0.734 ± 0.076 |
| breast | FedGA (baseline) | 5 | 0.156 ± 0.349 | 0.091 ± 0.045 | 0.788 ± 0.051 | 0.027 ± 0.057 | 0.700 ± 0.095 | 0.706 ± 0.066 |
| breast | XFedGA | 5 | 0.160 ± 0.347 | 0.091 ± 0.053 | 0.784 ± 0.035 | 0.024 ± 0.052 | 0.727 ± 0.095 | 0.724 ± 0.070 |

Test checkpoints (validation-selected), mean ± s.d. over seeds of the per-run mean over test images. rho_after_all_layers: lower = more faithful. Deletion / insertion AUC are exploratory (context caveat).
