| Task / protocol | FedAvg | FedGA | XFedGA | XFedGA+ | FedGA paper GA | XFedGA acc. | XFedGA+ acc. | Clean measurement? |
|---|---|---|---|---|---|---|---|---|
| breast v1 | 67 s | 70 s | 133 s | 157 s (derived) | 82 s (derived) | 104 s (derived) | 129 s (derived) | fedga: yes, xfedga: yes |
| breast v2 | 42 s (derived) | 49 s | 223 s | 257 s (derived) | 80 s (derived) | 142 s (derived) | 176 s (derived) | fedga: yes, xfedga: NO |
| prostate_ms v1 | 101 s (derived) | 104 s (derived) | 163 s (derived) | 186 s (derived) | 116 s (derived) | 138 s (derived) | 161 s (derived) | fedga: yes, xfedga: yes |
| prostate_ms v2 | 29 s (derived) | 36 s (derived) | 198 s (derived) | 250 s (derived) | 64 s (derived) | 126 s (derived) | 178 s (derived) | fedga: yes, xfedga: NO |

| Scenario | screens | final_headline | headline_xplus | headline_xplus_dsc | acceptance_screen | headline_accept | dose_p050_p100 | dose_p025 | fedga_paper | h_rob2 | ablations | parity | fedprox_bounds | Total (×1.2, h) | Capacity (h) | Cuts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overnight: breast v1, prostate_ms v1, XFedGA+ not adopted, acceptance not adopted | 16.9 | 9.3 | 0.0 | 0.0 | 2.4 | 0.0 | 8.6 | 4.3 | 4.1 | 2.1 | 19.0 | 2.4 | 9.6 | 78.5 | 144.6 | none |
| overnight: breast v1, prostate_ms v1, XFedGA+ not adopted, acceptance adopted | 16.9 | 9.3 | 0.0 | 0.0 | 2.4 | 6.2 | 8.6 | 4.3 | 4.1 | 2.1 | 19.0 | 2.4 | 9.6 | 84.7 | 144.6 | none |
| overnight: breast v1, prostate_ms v1, XFedGA+ adopted, acceptance not adopted | 16.9 | 9.3 | 8.8 | 8.8 | 2.9 | 0.0 | 13.4 | 6.7 | 4.1 | 2.1 | 19.0 | 2.4 | 9.6 | 103.8 | 144.6 | none |
| overnight: breast v1, prostate_ms v1, XFedGA+ adopted, acceptance adopted | 16.9 | 9.3 | 8.8 | 8.8 | 2.9 | 7.4 | 13.4 | 6.7 | 4.1 | 2.1 | 19.0 | 2.4 | 9.6 | 111.2 | 144.6 | none |
| overnight: breast v1, prostate_ms v2, XFedGA+ not adopted, acceptance not adopted | 16.9 | 6.6 | 0.0 | 0.0 | 2.3 | 0.0 | 8.6 | 4.3 | 3.0 | 2.1 | 18.5 | 2.4 | 5.6 | 70.2 | 144.6 | none |
| overnight: breast v1, prostate_ms v2, XFedGA+ not adopted, acceptance adopted | 16.9 | 6.6 | 0.0 | 0.0 | 2.3 | 5.9 | 8.6 | 4.3 | 3.0 | 2.1 | 18.5 | 2.4 | 5.6 | 76.1 | 144.6 | none |
| overnight: breast v1, prostate_ms v2, XFedGA+ adopted, acceptance not adopted | 16.9 | 6.6 | 10.3 | 10.3 | 3.1 | 0.0 | 13.4 | 6.7 | 3.0 | 2.1 | 18.5 | 2.4 | 5.6 | 99.0 | 144.6 | none |
| overnight: breast v1, prostate_ms v2, XFedGA+ adopted, acceptance adopted | 16.9 | 6.6 | 10.3 | 10.3 | 3.1 | 7.8 | 13.4 | 6.7 | 3.0 | 2.1 | 18.5 | 2.4 | 5.6 | 106.8 | 144.6 | none |

Per-round times: mean over runs (s); prostate_ms values are derived from the measured prostate components (scaling in the JSON). Not-clean measurements (overlap with other work on this machine): prostate_xfedga_seed1_2a059f3e_v2screen: GPU shared with a full pytest run (15:50-16:05) and slow tests (16:05-16:09); breast_fedga_seed1_019627d6_v2screen: 18 s GPU known-answer test; CPU CAM check; breast_xfedga_seed1_bb69b34e_v2screen: CPU-heavy multi-site leakage audit and breast duplicate audit; prostate_fedga_seed2_5666ea23_v2screen: CPU: prostate_ms cache build and CPU test runs; prostate_xfedga_seed2_2a059f3e_v2screen: CPU: EC tests, FOV-crop derivation, prostate_ms v1/v2 cache builds; breast_xfedga_seed2_bb69b34e_v2screen: GPU throttled: median SM clock 1335 MHz (queue log).
Capacity 'evenings': tonight 4 h (assumed) + the 17 Oct full days + 6 h on other days (assumed). Capacity 'overnight': tonight from the v2-screen end to midnight + full days unchanged + 16 h on other weekdays (20 h - 4 h buffer). To 2026-10-27.
