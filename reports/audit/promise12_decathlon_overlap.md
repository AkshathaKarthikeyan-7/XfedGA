# PROMISE12 (clients) vs Decathlon Task05 (server): duplicate cases

Source: `analysis/promise12_decathlon_overlap.py` → `results/audit/promise12_decathlon_overlap.json`. Images and masks only; no results read.

Duplicates: **12** pairs; PROMISE12 cases involved: 12 of 50.

| Decathlon server role | Cases in role | Duplicated by a PROMISE12 training case | Cases |
|---|---|---|---|
| fitness | 8 | 2 | prostate_01, prostate_14 |
| validation | 8 | 2 | prostate_07, prostate_16 |
| test | 16 | 4 | prostate_00, prostate_13, prostate_17, prostate_18 |
| not used (imagesTs) | — | 4 | prostate_05, prostate_09, prostate_11, prostate_15 |

| PROMISE12 case | Client | Decathlon case | Role | r (full res.) | Mask Dice |
|---|---|---|---|---|---|
| Case13 | PROMISE12_client1 | Decathlon_imagesTr/prostate_00 | test | 1.0 | 0.9535 |
| Case14 | PROMISE12_client2 | Decathlon_imagesTr/prostate_01 | fitness | 1.0 | 0.9845 |
| Case15 | PROMISE12_client1 | Decathlon_imagesTs/prostate_05 | not used (imagesTs) | 1.0 | None |
| Case17 | PROMISE12_client1 | Decathlon_imagesTr/prostate_07 | validation | 1.0 | 0.8904 |
| Case18 | PROMISE12_client2 | Decathlon_imagesTs/prostate_09 | not used (imagesTs) | 1.0 | None |
| Case19 | PROMISE12_client1 | Decathlon_imagesTs/prostate_11 | not used (imagesTs) | 1.0 | None |
| Case20 | PROMISE12_client2 | Decathlon_imagesTr/prostate_13 | test | 1.0 | 0.952 |
| Case21 | PROMISE12_client2 | Decathlon_imagesTr/prostate_14 | fitness | 1.0 | 0.9494 |
| Case22 | PROMISE12_client2 | Decathlon_imagesTs/prostate_15 | not used (imagesTs) | 1.0 | None |
| Case23 | PROMISE12_client2 | Decathlon_imagesTr/prostate_16 | validation | 1.0 | 1.0 |
| Case24 | PROMISE12_client1 | Decathlon_imagesTr/prostate_17 | test | 1.0 | 0.9264 |
| Case25 | PROMISE12_client1 | Decathlon_imagesTr/prostate_18 | test | 1.0 | 0.9394 |

Screened candidates not confirmed: 0.
