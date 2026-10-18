Case-level paired differences (seeds averaged within case), 95% cluster-bootstrap CI over cases. PRIMARY = the pre-registered endpoint (XFedGA vs FedGA (baseline), test DSC); every other row is secondary; deletion/insertion AUC are exploratory: on real images the model uses surrounding anatomy, which blurring the salient pixels in place leaves intact. Sensitivity: paired t-interval and Wilcoxon signed-rank p; DISAGREE marks a different WIN/TIE/LOSE between the bootstrap and the t-interval.

| task | comparison | metric | endpoint | cases | mean diff. | 95% CI (cluster bootstrap) | $d_z$ | outcome | 95% paired t-interval | t outcome | Wilcoxon p | bootstrap vs t |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prostate | XFedGA vs FedGA (baseline) | dsc | PRIMARY | 16 | +0.0149 | [-0.0019, +0.0327] | +0.41 | TIE | [-0.0046, +0.0344] | TIE | 0.13 |  |
| prostate | XFedGA vs FedGA (baseline) | hd95_norm | secondary | 16 | -0.0303 | [-0.0466, -0.0144] | -0.89 | WIN | [-0.0484, -0.0122] | WIN | 0.00336 |  |
| prostate | XFedGA vs FedAvg | dsc | secondary | 16 | +0.0065 | [-0.0095, +0.0216] | +0.20 | TIE | [-0.0110, +0.0240] | TIE | 0.323 |  |
| prostate | XFedGA vs FedAvg | hd95_norm | secondary | 16 | -0.0161 | [-0.0288, -0.0024] | -0.58 | WIN | [-0.0310, -0.0012] | WIN | 0.0182 |  |
| prostate | FedGA (baseline) vs FedAvg | dsc | secondary | 16 | -0.0084 | [-0.0265, +0.0119] | -0.20 | TIE | [-0.0302, +0.0135] | TIE | 0.298 |  |
| prostate | FedGA (baseline) vs FedAvg | hd95_norm | secondary | 16 | +0.0142 | [-0.0045, +0.0332] | +0.35 | TIE | [-0.0072, +0.0355] | TIE | 0.193 |  |
| breast | XFedGA vs FedGA (baseline) | dsc | PRIMARY | 638 | -0.0072 | [-0.0109, -0.0035] | -0.15 | LOSE | [-0.0109, -0.0035] | LOSE | 0.000944 |  |
| breast | XFedGA vs FedGA (baseline) | hd95_norm | secondary | 638 | +0.0115 | [+0.0083, +0.0147] | +0.28 | LOSE | [+0.0083, +0.0147] | LOSE | 2.92e-12 |  |
| breast | XFedGA vs FedAvg | dsc | secondary | 638 | -0.0034 | [-0.0064, -0.0004] | -0.09 | LOSE | [-0.0065, -0.0004] | LOSE | 0.0244 |  |
| breast | XFedGA vs FedAvg | hd95_norm | secondary | 638 | -0.0000 | [-0.0030, +0.0029] | -0.00 | TIE | [-0.0030, +0.0029] | TIE | 0.541 |  |
| breast | FedGA (baseline) vs FedAvg | dsc | secondary | 638 | +0.0038 | [-0.0001, +0.0075] | +0.08 | TIE | [-0.0001, +0.0076] | TIE | 0.0566 |  |
| breast | FedGA (baseline) vs FedAvg | hd95_norm | secondary | 638 | -0.0115 | [-0.0147, -0.0083] | -0.28 | WIN | [-0.0148, -0.0083] | WIN | 1.13e-15 |  |
