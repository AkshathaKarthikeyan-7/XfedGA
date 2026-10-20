Outcome (spec 9B): XFedGA vs FedGA (baseline).

| task | accuracy | explanation (independent metrics) | stability | cost |
|---|---|---|---|---|
| breast | LOSE (ΔDSC -0.007, CI [-0.011, -0.004]) | pg_gradcampp: TIE (+0.004, CI [-0.005, +0.013]); rho_after_all_layers: TIE (+0.000, CI [-0.002, +0.002]) | seed s.d. 0.014 vs 0.016; last-5-round val s.d. 0.013 vs 0.008 | 133 s vs 70 s per round (+89%) |
| prostate_ms | LOSE (ΔDSC -0.005, CI [-0.008, -0.001]) | pg_gradcampp: TIE (+0.014, CI [+0.000, +0.043]); rho_after_all_layers: LOSE (+0.021, CI [+0.009, +0.031]) | seed s.d. 0.003 vs 0.003; last-5-round val s.d. 0.005 vs 0.004 | 197 s vs 49 s per round (+301%) |
