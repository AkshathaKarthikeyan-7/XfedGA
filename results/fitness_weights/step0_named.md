| Task | Weights (w1, w2, w3) | Rounds | Winner changes vs (0.5, 0.3, 0.2) | Winner DSC (fitness subset) | Winner HD95_norm | Winner EC | Winner = w_agg |
|---|---|---|---|---|---|---|---|
| breast | (0.5, 0.3, 0.2) | 75 | 0 (0%) | 0.6559 | 0.2783 | 0.2193 | 1% |
| breast | (0.7, 0.2, 0.1) | 75 | 21 (28%) | 0.6567 | 0.2791 | 0.2173 | 1% |
| breast | (0.8, 0.2, 0.0) | 75 | 34 (45%) | 0.6572 | 0.2793 | 0.2126 | 0% |
| breast | (1.0, 0.0, 0.0) | 75 | 51 (68%) | 0.6576 | 0.2833 | 0.2115 | 4% |
| prostate | (0.5, 0.3, 0.2) | 75 | 0 (0%) | 0.4691 | 0.2049 | 0.1901 | 1% |
| prostate | (0.7, 0.2, 0.1) | 75 | 19 (25%) | 0.4705 | 0.2082 | 0.1899 | 1% |
| prostate | (0.8, 0.2, 0.0) | 75 | 23 (31%) | 0.4707 | 0.2090 | 0.1898 | 1% |
| prostate | (1.0, 0.0, 0.0) | 75 | 43 (57%) | 0.4715 | 0.2158 | 0.1880 | 3% |

Re-ranking of the logged candidate pools of the v1 headline XFedGA runs (CPU, no new evaluation). Within-round selection only: the GA trajectory under other weights is not simulated. Winner metrics are on the 40-image fitness subset, not validation.
