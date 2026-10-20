Final headline: 4/35 runs done; about 21.9 h left; ETA Tue 20 Oct 14:58. Capacity from Tue 20 Oct 16:43 to 27 Oct 24:00 (overnight mode): 112.9 GPU-h.

| Item | Hours (×1.2) | After cuts |
|---|---|---|
| XAI metrics on the 35 final checkpoints (no training; ~3 min per run measured 18 Oct) | 2.1 | 2.1 |
| a. dose-response breast p = 0.5, 1.0 x 4 methods x 3 seeds | 14.7 | 14.7 |
| a. dose-response breast p = 0.25 x 4 methods x 3 seeds | 7.4 | 7.4 |
| b. fl_dist breast XFedGA+ and XFedGA, seed 1 (demo) | 2.0 | 2.0 |
| b. fl_dist breast XFedGA+ and XFedGA, seeds 2-3 (parity) | 3.9 | 3.9 |
| c. FedGA paper GA settings (P 10, G 20, 20 rounds), 2 tasks x 3 seeds | 3.3 | 3.3 |
| d. H-ROB2 noisy-q_i client, breast, XFedGA and XFedGA+ x 3 seeds | 5.1 | 5.1 |
| e. ablations B, C, D, F + random search, 2 tasks x 3 seeds | 18.5 | 18.5 |
| f. FedProx 2 tasks x 5 seeds; centralized + local-only 2 tasks x 3 seeds | 6.9 | 6.9 |

Total a–f after cuts: 61.8 h of 112.9 h → fits. Cuts applied: none.
Derived (not measured): ; ablations, FedProx, bounds, paper-GA and fl_dist costs are derived from measured components (see the script docstring).
