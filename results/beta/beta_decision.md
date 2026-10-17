| task | beta | seeds | val score mean | seed s.d. | margin over beta 0.5 | beats 0.5 by > pooled s.d. | q_i variance |
|---|---|---|---|---|---|---|---|
| prostate | 0.0 | [1, 2] | 0.3689 | 0.0033 | -0.0225 | False | 7.82e-03 |
| prostate | 0.2 | [1, 2] | 0.3901 | 0.0016 | -0.0012 | False | 4.20e-03 |
| prostate | 0.5 **(selected)** | [1, 2] | 0.3914 | 0.0128 | +0.0000 | False | 6.73e-03 |
| prostate | 1.0 | [1, 2] | 0.3575 | 0.0279 | -0.0339 | False | 8.13e-03 |
| prostate | pooled s.d. = 0.0155 | | | | | default beta 0.5 kept: no beta beats it by more than the pooled s.d. | |
| breast | 0.0 | [1, 2] | 0.6558 | 0.0005 | -0.0129 | False | 3.04e-03 |
| breast | 0.2 | [1, 2] | 0.6658 | 0.0028 | -0.0029 | False | 2.20e-03 |
| breast | 0.5 **(selected)** | [1, 2] | 0.6687 | 0.0082 | +0.0000 | False | 2.10e-03 |
| breast | 1.0 | [1, 2] | 0.6491 | 0.0040 | -0.0197 | False | 2.10e-03 |
| breast | pooled s.d. = 0.0048 | | | | | default beta 0.5 kept: no beta beats it by more than the pooled s.d. | |

Margins vs other betas are EXPLORATORY (2 seeds; prostate 8 / breast 267 validation cases; validation only). The screen selects beta; it is not evidence that q_i weighting helps.
