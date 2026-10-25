### b. Flower parity (breast; fl_dist vs custom loop, seeds 1–3; TOST ±0.02)

| Method | Patients | Δ test DSC (fl − custom) | 90% CI | Equivalent | Max \|Δ val DSC per round\| | Message audit |
|---|---|---|---|---|---|---|
| XFedGA | 638 | +0.0000 | [+0.0000, +0.0000] | yes | 0.00e+00 | ok |
| XFedGA+ | 638 | +0.0000 | [+0.0000, +0.0000] | yes | 0.00e+00 | ok |

### c. FedGA with the paper's GA settings (seeds 1–3)

| Task | Comparison | Role | Cases | Δ test DSC | 95% CI | Outcome | t-interval |
|---|---|---|---|---|---|---|---|
| breast | XFedGA vs FedGA (paper GA settings) | secondary | 638 | -0.0104 | [-0.0143, -0.0064] | **LOSE** | LOSE |
| breast | XFedGA+ vs FedGA (paper GA settings) | secondary | 638 | +0.0028 | [-0.0039, +0.0095] | **TIE** | TIE |
| breast | FedGA (paper GA settings) vs FedGA (baseline) | secondary | 638 | -0.0008 | [-0.0047, +0.0032] | **TIE** | TIE |
| prostate_ms | XFedGA vs FedGA (paper GA settings) | secondary | 14 | -0.0091 | [-0.0168, -0.0027] | **LOSE** | LOSE |
| prostate_ms | XFedGA+ vs FedGA (paper GA settings) | secondary | 14 | +0.0027 | [-0.0043, +0.0105] | **TIE** | TIE |
| prostate_ms | FedGA (paper GA settings) vs FedGA (baseline) | secondary | 14 | +0.0008 | [-0.0040, +0.0060] | **TIE** | TIE |

### d. H-ROB2: noisy q_i (BUS-UCLM reports q_i = 1), breast, seeds 1–3

| Method | Role | Patients | Drop (clean − noisy) | 95% CI | Supported (inside ±0.02) |
|---|---|---|---|---|---|
| XFedGA | primary of d | 638 | -0.0153 | [-0.0208, -0.0098] | no |
| XFedGA+ | secondary | 638 | +0.0000 | [+0.0000, +0.0000] | yes |

### e. Ablations (seeds 1–3)

| Task | Comparison | Role | Cases | Δ test DSC | 95% CI | Outcome | t-interval |
|---|---|---|---|---|---|---|---|
| breast | B: q_i weighting only vs FedAvg | secondary | 638 | +0.0021 | [-0.0016, +0.0057] | **TIE** | TIE |
| breast | C: FedAvg + GA (multi) vs FedGA (baseline) | secondary | 638 | -0.0134 | [-0.0183, -0.0086] | **LOSE** | LOSE |
| breast | D: FedGA + annealing vs FedGA (baseline) | secondary | 638 | -0.0042 | [-0.0078, -0.0006] | **LOSE** | LOSE |
| breast | C: FedAvg + GA (multi) vs random search | secondary | 638 | -0.0099 | [-0.0169, -0.0027] | **LOSE** | LOSE |
| breast | XFedGA vs F: XFedGA, random q_i | secondary | 638 | -0.0027 | [-0.0066, +0.0012] | **TIE** | TIE |
| breast | XFedGA vs C: FedAvg + GA (multi) | secondary | 638 | +0.0023 | [-0.0022, +0.0068] | **TIE** | TIE |
| prostate_ms | B: q_i weighting only vs FedAvg | secondary | 14 | -0.0031 | [-0.0120, +0.0047] | **TIE** | TIE |
| prostate_ms | C: FedAvg + GA (multi) vs FedGA (baseline) | secondary | 14 | -0.0058 | [-0.0098, -0.0017] | **LOSE** | LOSE |
| prostate_ms | D: FedGA + annealing vs FedGA (baseline) | secondary | 14 | -0.0067 | [-0.0107, -0.0035] | **LOSE** | LOSE |
| prostate_ms | C: FedAvg + GA (multi) vs random search | secondary | 14 | -0.0011 | [-0.0046, +0.0024] | **TIE** | TIE |
| prostate_ms | XFedGA vs F: XFedGA, random q_i | secondary | 14 | -0.0122 | [-0.0181, -0.0068] | **LOSE** | LOSE |
| prostate_ms | XFedGA vs C: FedAvg + GA (multi) | secondary | 14 | -0.0025 | [-0.0074, +0.0017] | **TIE** | TIE |

### f. FedProx (seeds 1–5)

| Task | Comparison | Role | Cases | Δ test DSC | 95% CI | Outcome | t-interval |
|---|---|---|---|---|---|---|---|
| breast | XFedGA vs FedProx | secondary | 638 | +0.0609 | [+0.0539, +0.0681] | **WIN** | WIN |
| breast | XFedGA+ vs FedProx | secondary | 638 | +0.0603 | [+0.0515, +0.0691] | **WIN** | WIN |
| breast | FedProx vs FedAvg | secondary | 638 | -0.0643 | [-0.0717, -0.0571] | **LOSE** | LOSE |
| prostate_ms | XFedGA vs FedProx | secondary | 14 | +0.0140 | [+0.0070, +0.0219] | **WIN** | WIN |
| prostate_ms | XFedGA+ vs FedProx | secondary | 14 | +0.0247 | [+0.0124, +0.0378] | **WIN** | WIN |
| prostate_ms | FedProx vs FedAvg | secondary | 14 | -0.0199 | [-0.0307, -0.0097] | **LOSE** | LOSE |

**Bounds (descriptive; seeds 1–3; test DSC mean ± s.d. over seeds):**

| Task | Bound | Test DSC |
|---|---|---|
| breast | centralized (pooled clients) | 0.6442 ± 0.0090 |
| breast | local-only, client BUSI | 0.6005 ± 0.0268 |
| breast | local-only, client BUS-UCLM | 0.5402 ± 0.0230 |
| prostate_ms | centralized (pooled clients) | 0.8476 ± 0.0084 |
| prostate_ms | local-only, client BIDMC | 0.4991 ± 0.0577 |
| prostate_ms | local-only, client HK | 0.6032 ± 0.0278 |
| prostate_ms | local-only, client I2CVB | 0.5755 ± 0.0126 |
| prostate_ms | local-only, client RUNMC | 0.7722 ± 0.0342 |
| prostate_ms | local-only, client UCL | 0.7548 ± 0.0185 |
