| Method | test DSC, p = 0.0 | test DSC, p = 0.25 | test DSC, p = 0.5 | test DSC, p = 1.0 | mean slope (DSC per unit p) |
|---|---|---|---|---|---|
| FedAvg | 0.6381 | 0.6123 | 0.6229 | 0.6016 | -0.0304 |
| FedGA (baseline) | 0.6411 | 0.6281 | 0.6045 | 0.6076 | -0.0341 |
| XFedGA | 0.6300 | 0.6261 | 0.6181 | 0.6160 | -0.0145 |
| XFedGA+ | 0.6432 | 0.6180 | 0.6220 | 0.6092 | -0.0287 |

| Test | Comparison | Patients | Δ slope | 95% CI (cluster bootstrap) | Supported (CI > 0) |
|---|---|---|---|---|---|
| D1 primary | XFedGA − FedGA (baseline) | 638 | +0.0197 | [+0.0145, +0.0247] | yes |
| D1 secondary | XFedGA+ − FedGA (baseline) | 638 | +0.0054 | [-0.0007, +0.0116] | no |

| D2 (mechanism) | alpha of BUS-UCLM at p = 0.0, 0.25, 0.5, 1.0 | OLS slope | 95% t-interval | Supported (< 0) |
|---|---|---|---|---|
| XFedGA | 0.2917, 0.2744, 0.2643, 0.2233 | -0.0676 | [-0.0969, -0.0382] | yes |
| XFedGA+ | 0.3042, 0.2466, 0.2177, 0.1313 | -0.1679 | [-0.1851, -0.1507] | yes |

Breast (v1), seeds 1–3; p = 0 reused (v1 headline; XFedGA+ final runs). Test evaluated once per run. Pre-registration (a), commit 4e51538.
