# Breast beta screen (validation only; for the slides)

Source: `results/beta/beta_decision.json` (`analysis/beta_decision.py`, rule of `b556334`; breast rows only — the prostate rows of that screen are SUPERSEDED by the PROMISE12 → Decathlon leakage).

| beta | Val score: mean | seed s.d. | − beta 0.5 | beats 0.5 by > pooled s.d. |
|---|---|---|---|---|
| 0.0 | 0.6558 | 0.0005 | -0.0129 | False |
| 0.2 | 0.6658 | 0.0028 | -0.0029 | False |
| 0.5 **(selected)** | 0.6687 | 0.0082 | +0.0000 | False |
| 1.0 | 0.6491 | 0.0040 | -0.0197 | False |

Pooled s.d. = 0.0048. Decision: default beta 0.5 kept: no beta beats it by more than the pooled s.d.. Margins are exploratory (2 seeds).
