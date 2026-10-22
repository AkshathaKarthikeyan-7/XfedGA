# Fitness-weight sensitivity (w1, w2, w3), breast

*For the panel question on the fitness weights (called lambda1-3 on the panel; w1, w2, w3 here): F = w1*DSC - w2*HD95_norm + w3*EC, default (0.5, 0.3, 0.2).*

**Method (Step 0, offline):** the logged GA candidate pools of the 5 v1 breast-headline XFedGA runs (75 rounds) are re-ranked under other weights (`analysis/fitness_weight_step0.py` -> `results/fitness_weights/step0.json`). No new training. The superseded prostate rows are excluded.

| (w1, w2, w3) | Rounds | GA winner changes vs (0.5, 0.3, 0.2) | Winner DSC (fitness subset) | Winner HD95_norm | Winner EC |
|---|---|---|---|---|---|
| (0.5, 0.3, 0.2) | 75 | 0 (0%) | 0.6559 | 0.2783 | 0.2193 |
| (0.7, 0.2, 0.1) | 75 | 21 (28%) | 0.6567 | 0.2791 | 0.2173 |
| (0.8, 0.2, 0.0) | 75 | 34 (45%) | 0.6572 | 0.2793 | 0.2126 |
| (1.0, 0.0, 0.0) | 75 | 51 (68%) | 0.6576 | 0.2833 | 0.2115 |

Over the whole 0.1-step simplex the winner changes in 48% of rounds on average; at the corners (1,0,0) 68%, (0,1,0) 69%, (0,0,1) 79%.

**Reading:** the weights change WHICH candidate wins in many rounds, but the winner's fitness-subset DSC moves very little (table). Caveat: within-round selection only; the GA trajectory under other weights is not simulated, and winner metrics are on the fitness subset, not validation. A validation-only weight screen on `prostate_ms` (item h, plan.md) tests the trajectory effect: `results/sensitivity/fw_screen_prostate_ms.md`.
