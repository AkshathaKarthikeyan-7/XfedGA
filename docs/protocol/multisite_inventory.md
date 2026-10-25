# Multi-site prostate: inventory and leakage audit

Source: `analysis/multisite_inventory.py` → `results/audit/multisite_inventory.json`. CPU only, no training.

| Site | Cases | Slices (min–max; total) | Slices with gland | In-plane size | Spacing x×y (mm) | Spacing z (mm) | Mask labels |
|---|---|---|---|---|---|---|---|
| BIDMC | 12 | 34–54; 521 | 261 | 384x384 | 0.365x0.365, 0.417x0.417 | 2.2, 2.8, 3.0 | [0, 1] |
| BMC | 30 | 23–38; 966 | 384 | 384x384 | 0.417x0.417, 0.443x0.443 | 3.0, 4.0 | [0, 1, 2] |
| HK | 12 | 24–24; 288 | 158 | 384x384 | 0.521x0.521 | 3.6 | [0, 1] |
| I2CVB | 19 | 64–64; 1216 | 468 | 384x384 | 0.534x0.401, 0.534x0.428, 0.534x0.429 … | 1.25 | [0, 1] |
| RUNMC | 30 | 15–24; 578 | 421 | 384x384 | 0.500x0.500, 0.521x0.521 | 3.6, 4.0 | [0, 1, 2] |
| UCL | 13 | 23–29; 318 | 175 | 384x384 | 0.469x0.469, 0.521x0.521 | 3.0, 3.3, 3.33, 3.6 | [0, 1] |

References: PROMISE12 50 cases (our clients); Decathlon imagesTr 32 (our server sets); Decathlon imagesTs 16 (unused).

| Site | Cases duplicating PROMISE12 / Decathlon | Of which sets | Cross-site duplicates | Clean vs PROMISE12 + Decathlon |
|---|---|---|---|---|
| BIDMC | 12 of 12 | PROMISE12 | 0 | no |
| BMC | 0 of 30 | — | 0 | yes |
| HK | 12 of 12 | PROMISE12 | 0 | no |
| I2CVB | 0 of 19 | — | 0 | yes |
| RUNMC | 30 of 30 | Decathlon_imagesTr, Decathlon_imagesTs, PROMISE12 | 0 | no |
| UCL | 13 of 13 | PROMISE12 | 0 | no |

Per-case duplicates (case → reference: full-resolution r, mask Dice):

- BIDMC/Case00 → PROMISE12/Case00 (r=0.99871, Dice=0.9937)
- BIDMC/Case02 → PROMISE12/Case02 (r=0.99867, Dice=0.9921)
- BIDMC/Case03 → PROMISE12/Case03 (r=0.99788, Dice=0.9919)
- BIDMC/Case04 → PROMISE12/Case04 (r=0.99882, Dice=0.9922)
- BIDMC/Case05 → PROMISE12/Case05 (r=0.99759, Dice=0.9939)
- BIDMC/Case06 → PROMISE12/Case06 (r=0.99849, Dice=0.9945)
- BIDMC/Case07 → PROMISE12/Case07 (r=0.99817, Dice=0.9938)
- BIDMC/Case08 → PROMISE12/Case08 (r=0.99868, Dice=0.9912)
- BIDMC/Case09 → PROMISE12/Case09 (r=0.99851, Dice=0.9939)
- BIDMC/Case10 → PROMISE12/Case10 (r=0.99888, Dice=0.9925)
- BIDMC/Case11 → PROMISE12/Case11 (r=0.99838, Dice=0.9932)
- BIDMC/Case12 → PROMISE12/Case12 (r=0.99894, Dice=0.9931)
- HK/Case38 → PROMISE12/Case38 (r=0.9992, Dice=0.9866)
- HK/Case39 → PROMISE12/Case39 (r=0.99916, Dice=0.9834)
- HK/Case40 → PROMISE12/Case40 (r=0.99926, Dice=0.9792)
- HK/Case41 → PROMISE12/Case41 (r=0.99919, Dice=0.9822)
- HK/Case42 → PROMISE12/Case42 (r=0.99908, Dice=0.9832)
- HK/Case43 → PROMISE12/Case43 (r=0.99923, Dice=0.9823)
- HK/Case44 → PROMISE12/Case44 (r=0.99918, Dice=0.981)
- HK/Case45 → PROMISE12/Case45 (r=0.9991, Dice=0.976)
- HK/Case46 → PROMISE12/Case46 (r=0.99895, Dice=0.9847)
- HK/Case47 → PROMISE12/Case47 (r=0.99909, Dice=0.981)
- HK/Case48 → PROMISE12/Case48 (r=0.99915, Dice=0.9833)
- HK/Case49 → PROMISE12/Case49 (r=0.99927, Dice=0.9854)
- RUNMC/Case00 → PROMISE12/Case13 (r=0.99922, Dice=0.9335); Decathlon_imagesTr/prostate_00 (r=0.99922, Dice=0.9452)
- RUNMC/Case01 → PROMISE12/Case14 (r=0.9988, Dice=0.9364); Decathlon_imagesTr/prostate_01 (r=0.9988, Dice=0.9284)
- RUNMC/Case02 → Decathlon_imagesTr/prostate_02 (r=0.99889, Dice=0.9536)
- RUNMC/Case03 → Decathlon_imagesTs/prostate_03 (r=0.99892, Dice=None)
- RUNMC/Case04 → PROMISE12/Case15 (r=0.9991, Dice=0.9438); Decathlon_imagesTs/prostate_05 (r=0.9991, Dice=None)
- RUNMC/Case05 → Decathlon_imagesTr/prostate_06 (r=0.99884, Dice=0.8949)
- RUNMC/Case06 → PROMISE12/Case17 (r=0.99848, Dice=0.9177); Decathlon_imagesTr/prostate_07 (r=0.99848, Dice=0.9008)
- RUNMC/Case07 → Decathlon_imagesTs/prostate_08 (r=0.99926, Dice=None)
- RUNMC/Case08 → PROMISE12/Case18 (r=0.99934, Dice=0.9473); Decathlon_imagesTs/prostate_09 (r=0.99934, Dice=None)
- RUNMC/Case09 → Decathlon_imagesTr/prostate_10 (r=0.99908, Dice=0.8396)
- RUNMC/Case10 → PROMISE12/Case19 (r=0.99908, Dice=0.9043); Decathlon_imagesTs/prostate_11 (r=0.99908, Dice=None)
- RUNMC/Case11 → Decathlon_imagesTs/prostate_12 (r=0.99931, Dice=None)
- RUNMC/Case12 → Decathlon_imagesTr/prostate_04 (r=0.9994, Dice=0.9306)
- RUNMC/Case13 → PROMISE12/Case20 (r=0.99889, Dice=0.8765); Decathlon_imagesTr/prostate_13 (r=0.99889, Dice=0.8628)
- RUNMC/Case14 → PROMISE12/Case21 (r=0.99893, Dice=0.9285); Decathlon_imagesTr/prostate_14 (r=0.99893, Dice=0.9099)
- RUNMC/Case15 → PROMISE12/Case22 (r=0.99885, Dice=0.9393); Decathlon_imagesTs/prostate_15 (r=0.99885, Dice=None)
- RUNMC/Case16 → PROMISE12/Case23 (r=0.99876, Dice=0.9834); Decathlon_imagesTr/prostate_16 (r=0.99876, Dice=0.9861)
- RUNMC/Case17 → PROMISE12/Case24 (r=0.99895, Dice=0.9333); Decathlon_imagesTr/prostate_17 (r=0.99895, Dice=0.8958)
- RUNMC/Case18 → PROMISE12/Case25 (r=0.99553, Dice=0.9583); Decathlon_imagesTr/prostate_18 (r=0.99553, Dice=0.9389)
- RUNMC/Case19 → Decathlon_imagesTs/prostate_19 (r=0.99891, Dice=None)
- RUNMC/Case20 → Decathlon_imagesTr/prostate_20 (r=0.99875, Dice=0.9452)
- RUNMC/Case21 → Decathlon_imagesTr/prostate_21 (r=0.99909, Dice=0.9003)
- RUNMC/Case22 → Decathlon_imagesTs/prostate_23 (r=0.99883, Dice=None)
- RUNMC/Case23 → Decathlon_imagesTr/prostate_24 (r=0.99893, Dice=0.9575)
- RUNMC/Case24 → Decathlon_imagesTr/prostate_25 (r=0.99874, Dice=0.918)
- RUNMC/Case25 → Decathlon_imagesTs/prostate_26 (r=0.99895, Dice=None)
- RUNMC/Case26 → Decathlon_imagesTs/prostate_27 (r=0.99904, Dice=None)
- RUNMC/Case27 → Decathlon_imagesTr/prostate_29 (r=0.99934, Dice=0.955)
- RUNMC/Case28 → Decathlon_imagesTr/prostate_31 (r=0.99899, Dice=0.8941)
- RUNMC/Case29 → Decathlon_imagesTr/prostate_44 (r=0.999, Dice=0.8701)
- UCL/Case01 → PROMISE12/Case01 (r=0.99875, Dice=0.992)
- UCL/Case26 → PROMISE12/Case26 (r=0.99915, Dice=0.9925)
- UCL/Case27 → PROMISE12/Case27 (r=0.99935, Dice=0.9915)
- UCL/Case28 → PROMISE12/Case28 (r=0.99931, Dice=0.9919)
- UCL/Case29 → PROMISE12/Case29 (r=0.99923, Dice=0.9932)
- UCL/Case30 → PROMISE12/Case30 (r=0.99923, Dice=0.9929)
- UCL/Case31 → PROMISE12/Case31 (r=0.99903, Dice=0.9911)
- UCL/Case32 → PROMISE12/Case32 (r=0.9993, Dice=0.9898)
- UCL/Case33 → PROMISE12/Case33 (r=0.99816, Dice=0.9921)
- UCL/Case34 → PROMISE12/Case34 (r=1.0, Dice=1.0)
- UCL/Case35 → PROMISE12/Case35 (r=0.99431, Dice=0.9817)
- UCL/Case36 → PROMISE12/Case36 (r=0.99863, Dice=0.9907)
- UCL/Case37 → PROMISE12/Case37 (r=0.99901, Dice=0.982)

Screened candidates (r ≥ 0.9 at 96 px) that were NOT confirmed as duplicates: 0.
