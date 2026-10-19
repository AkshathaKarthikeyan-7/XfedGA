| Task (protocol) | Method | Val score: mean (seeds 1, 2) |
|---|---|---|
| breast (v1) | FedGA (baseline) | 0.6667 (0.6713, 0.6622) |
| breast (v1) | XFedGA | 0.6687 (0.6745, 0.6629) |
| breast (v1) | XFedGA+ | 0.6799 (0.6799, 0.6798) |
| breast (v1) | XFedGA+ (DSC-only) | 0.6747 (0.6796, 0.6698) |
| breast | XFedGA+ − XFedGA = +0.0111; pooled s.d. = 0.0063 | beats: True; worse: False |
| prostate_ms (v2) | FedGA (baseline) | 0.8401 (0.8387, 0.8416) |
| prostate_ms (v2) | XFedGA | 0.8331 (0.8368, 0.8295) |
| prostate_ms (v2) | XFedGA+ | 0.8495 (0.8513, 0.8476) |
| prostate_ms (v2) | XFedGA+ (DSC-only) | 0.8491 (0.8508, 0.8475) |
| prostate_ms | XFedGA+ − XFedGA = +0.0163; pooled s.d. = 0.0033 | beats: True; worse: False |

**Decision: adopt XFedGA+.** Validation only; XFedGA+ is a declared post-v1 variant.
