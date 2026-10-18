Outcome (spec 9B): XFedGA vs FedGA (baseline).

| task | accuracy | explanation (independent metrics) | stability | cost |
|---|---|---|---|---|
| prostate | TIE (ΔDSC +0.015, CI [-0.002, +0.033]) | pg_gradcampp: TIE (+0.001, CI [+0.000, +0.002]); rho_after_all_layers: LOSE (+0.028, CI [+0.022, +0.032]) | seed s.d. 0.041 vs 0.070; last-5-round val s.d. 0.027 vs 0.024 | 114 s vs 58 s per round (+95%) |
| breast | LOSE (ΔDSC -0.007, CI [-0.011, -0.004]) | pg_gradcampp: TIE (+0.004, CI [-0.005, +0.013]); rho_after_all_layers: TIE (+0.000, CI [-0.002, +0.002]) | seed s.d. 0.014 vs 0.016; last-5-round val s.d. 0.013 vs 0.008 | 133 s vs 70 s per round (+89%) |
