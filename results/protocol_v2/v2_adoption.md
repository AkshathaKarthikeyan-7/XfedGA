| Task | Method | v1 score (seeds 1, 2) | v2 score (seeds 1, 2) | v2 − v1 | Beats by > pooled s.d. |
|---|---|---|---|---|---|
| breast | FedGA (baseline) | 0.6667 (0.6713, 0.6622) | 0.6779 (0.6698, 0.6860) | +0.0112 | True |
| breast | XFedGA | 0.6687 (0.6745, 0.6629) | 0.6740 (0.6746, 0.6733) | +0.0052 | False |
| breast | pooled s.d. = 0.0077 | | | | **keep v1: v2 does not beat v1 by more than the pooled s.d. for XFedGA** |
| prostate_ms | FedGA (baseline) | 0.6948 (0.7233, 0.6664) | 0.8401 (0.8387, 0.8416) | +0.1453 | True |
| prostate_ms | XFedGA | 0.6668 (0.6624, 0.6712) | 0.8331 (0.8368, 0.8295) | +0.1663 | True |
| prostate_ms | pooled s.d. = 0.0205 | | | | **adopt v2: beats v1 by more than the pooled s.d. for both FedGA and XFedGA** |

Validation only. Score = max centred 3-round mean validation DSC per run, mean of seeds 1–2; each protocol is measured in its own pipeline (prostate_ms v2 crop, S = 110 mm, cuts no BMC fitness/validation gland pixels: results/audit/fov_crop_prostate_ms.json). The old task `prostate` (PROMISE12 -> Decathlon) is SUPERSEDED and not decided here.

| v2 run | Mean val DSC: GA winner | Mean val DSC: w_agg | Rounds winner < w_agg |
|---|---|---|---|
| breast_fedga_seed1_019627d6_v2screen | 0.6403 | 0.6349 | 13% |
| breast_fedga_seed2_019627d6_v2screen | 0.6520 | 0.6442 | 13% |
| breast_xfedga_seed1_bb69b34e_v2screen | 0.6405 | 0.6359 | 20% |
| breast_xfedga_seed2_bb69b34e_v2screen | 0.6476 | 0.6446 | 20% |
| prostate_ms_fedga_seed1_82c70851_msv1screen | 0.6383 | 0.6285 | 7% |
| prostate_ms_fedga_seed2_82c70851_msv1screen | 0.6053 | 0.5983 | 33% |
| prostate_ms_fedga_seed1_c83f1ad3_msv2screen | 0.7639 | 0.7507 | 13% |
| prostate_ms_fedga_seed2_c83f1ad3_msv2screen | 0.7629 | 0.7581 | 20% |
| prostate_ms_xfedga_seed1_45a979a4_msv1screen | 0.6137 | 0.6009 | 20% |
| prostate_ms_xfedga_seed2_45a979a4_msv1screen | 0.5966 | 0.5902 | 20% |
| prostate_ms_xfedga_seed1_ea92b08b_msv2screen | 0.7684 | 0.7600 | 13% |
| prostate_ms_xfedga_seed2_ea92b08b_msv2screen | 0.7558 | 0.7496 | 27% |
