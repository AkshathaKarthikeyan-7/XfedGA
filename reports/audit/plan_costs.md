| id | tier | prio | experiment | per-run h (prostate / breast) | runs per task | total GPU-h |
|---|---|---|---|---|---|---|
| T1-beta | 1 | 1b | beta ablation, beta in {0,0.2,0.25,0.5,1.0} | 0.83 / 0.95 | 15 | 26.7 |
| T1-flsmoke | 1 | 2 | fl_dist 1-round smoke run + message audit | 0.07 / 0.10 | 1 | 0.2 |
| T1-core-fedavg | 1 | 3 | FedAvg | 0.29 / 0.54 | 5 | 4.2 |
| T1-core-A | 1 | 3 | FedGA (baseline) | 0.34 / 0.59 | 5 | 4.7 |
| T1-core-E | 1 | 3 | XFedGA at selected beta (3 seeds reused from beta ablation) | 0.83 / 0.95 | 2 | 3.6 |
| T1-core-fedprox | 1 | 3* | FedProx (spec section 5 baseline; not named in tier list) | 0.29 / 0.54 | 5 | 4.2 |
| T1-parity | 1 | 4 | fl_dist XFedGA, breast (custom seeds reused) | – / 1.10 | 3 | 3.3 |
| T1-abl-B | 1 | 5 | B weighted aggregation only | 0.31 / 0.56 | 3 | 2.6 |
| T1-abl-F | 1 | 5 | F XFedGA with random q_i | 0.81 / 0.94 | 3 | 5.3 |
| T1-abl-C | 1 | 5c | C FedAvg + GA(multi-objective) | 0.81 / 0.94 | 3 | 5.3 |
| T1-abl-D | 1 | 5c | D FedGA + annealing | 0.34 / 0.59 | 3 | 2.8 |
| T1-abl-RS | 1 | 5c | random-search control (matched evaluations) | 0.81 / 0.94 | 3 | 5.3 |
| T1-fullfit | 1 | 5c | full-fitness-set check (A and E, 1 seed) | 2.33 / 4.22 | 1 | 6.5 |
| T1-bounds | 1 | 3* | centralized + local-only bounds | 0.60 / 1.14 | 3 | 5.2 |
| OPT-paper | opt | opt | FedGA (paper GA settings): P=10, G=20, 20 rounds, batch 8, 1 seed | 1.57 / 3.31 | 1 | 4.9 |
| T2-fw-step0 | 2 | 6 | fitness-weight Step 0: 1 run with saved candidate pools (offline re-rank) | 0.83 / 0.95 | 1 | 1.8 |
| T2-fw-corners | 2 | 6 | fitness-weight corners + 3 extra points (breast), transfer check (prostate) | 0.83 / 0.95 | 18 | 32.1 |
| T2-sens | 2 | 6 | sensitivity: pop {4,10}, gen {3,10}, sigma0 {0.005,0.02}, tau {1,2,5} (breast) | – / 0.95 | 27 | 25.7 |
| T2-hetero | 2 | 7 | heterogeneous prostate partition: FedAvg, A, E | 1.46 / – | 5 | 7.3 |
| T2-robust | 2 | 9 | robustness: q_i=1 client, corrupted-label client; E and FedAvg (breast) | – / 1.50 | 6 | 9.0 |
| T2-dp | 2 | 9 | DP: clipping + Gaussian noise, XFedGA (breast, optional) | – / 0.95 | 3 | 2.9 |
| T2-expl | 2 | 8 | NSD + independent explanation metrics (eval only, 20 checkpoints/task) | 0.09 / 0.09 | 20 | 3.7 |
