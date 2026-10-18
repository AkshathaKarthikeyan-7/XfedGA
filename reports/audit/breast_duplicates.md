# Breast duplicate / leakage audit

Source: `analysis/breast_duplicate_audit.py` → `results/audit/breast_duplicates.json`. CPU only, images only.

Images: BUSI 780, BUS-UCLM 683, BUS-BRA 1875.

| Pair group | Duplicates (r ≥ 0.98 at 256 px or identical pixels) | Near-duplicates (0.95–0.98) | Screened, not confirmed |
|---|---|---|---|
| BUS-BRA vs BUSI | 0 | 0 | 0 |
| BUS-BRA vs BUS-UCLM | 0 | 0 | 0 |
| BUS-UCLM vs BUSI | 0 | 0 | 0 |
| BUS-BRA different patients, across splits | 0 | 0 | 0 |
| BUS-BRA different patients, same split | 0 | 0 | 0 |
| BUS-BRA same patient | 0 | 0 | 0 |
| inside BUSI | 26 | 70 | 86 |
| inside BUS-UCLM | 8 | 9 | 4 |

**Train/server or cross-split leakage found: none.**

Limitation: a copy cropped or padded differently is not detected by whole-image correlation.
