# Breast results: Review 2 vs corrected

*Every number below is read from a file: the Review-2 deck (slide 10) or `results/` (`analysis/review2_vs_corrected.py` → `results/review/review2_vs_corrected.json`). No new runs, no test re-evaluation.*

## Review 2 (as presented; SUPERSEDED) — n = 937 OOD test images • Base FedGA vs XFedGA • per-image paired tests

| Metric | Base FedGA | XFedGA | Interpretation |
|---|---|---|---|
| DSC | 0.646 | 0.650 | Tied (p = 0.30) |
| HD95 (normalised) | 0.280 | 0.263 | Better (p < 0.0001) |
| EC (IoU@0.5) | 0.145 | 0.153 | Better (p = 0.0066) |
| Pointing game | 57.6% | 71.7% | Better (p < 0.0001) |

Review 2 setup: checkpoint = best round on the OOD **test** set; one unseeded run per method; per-image tests on correlated images; BUS-UCLM malignant masks dropped; BUS-BRA fitness / test split per image (patients in both); old Grad-CAM layer.

## Corrected (breast v1 headline; 1127 test images from 638 BUS-BRA patients; 5 seeds)

| Metric | FedGA (baseline) | XFedGA | XFedGA − FedGA (patient-level) | 95% CI (cluster bootstrap) | Outcome |
|---|---|---|---|---|---|
| DSC | 0.638 ± 0.014 | 0.631 ± 0.015 | -0.0072 | [-0.0109, -0.0035] | LOSE |
| HD95 (normalised) | 0.247 ± 0.014 | 0.258 ± 0.023 | +0.0115 | [+0.0083, +0.0147] | LOSE |
| EC (IoU@0.5) | 0.212 ± 0.081 | 0.223 ± 0.084 | +0.0110 | [+0.0085, +0.0137] | WIN |
| EC_mass | 0.446 ± 0.160 | 0.435 ± 0.158 | -0.0105 | [-0.0126, -0.0084] | LOSE |
| Pointing game | 80.0% ± 3.1% | 79.0% ± 2.5% | -1.2 pp | [-2.1, -0.2] pp | LOSE |

WIN / TIE / LOSE are from XFedGA's point of view (lower HD95 is better). Mean ± s.d. over the 5 seeds of the per-run mean over test images. The pointing game and EC here use the corrected Grad-CAM (full-resolution decoder layer, predicted-foreground target).

## Which Review-2 numbers each audit item affects

Items 1–9 follow `reports/audit/reaudit.md` (thesis Chapter 4 numbering, confirmed 18 Oct 2026). Slide numbers refer to the Review-2 deck.

| Item | Issue | Review-2 numbers affected |
|---|---|---|
| 1 | Checkpoint chosen as the best round on the OOD **test** set (`results/audit/verify_checkpoints.json`) | **All** accuracy and explanation numbers: breast table (slide 10: DSC, HD95, EC, pointing game), prostate table (slide 11), cross-task summary (slide 12), conclusions (slide 18). All optimistic. |
| 2 | Single unseeded runs; per-image tests on correlated images | Every p-value and every Tied / Better / Worse interpretation on slides 10 and 11 (n = 937 images, n = 233 slices); slide 7 test choices. Certainty overstated; no seed variance. |
| 3 | BUS-UCLM malignant (red) masks binarised to empty: all malignant lesions dropped (audit D-1) | Breast: every number on slide 10 (the BUS-UCLM client trained on benign lesions only); slide 9 BUS-UCLM empty-mask share and client size. |
| 4 | BUS-BRA fitness / OOD split per image: patients in both halves (audit D-2) | Breast: every number on slide 10; slide 9 split (938 fitness / 937 OOD) and slide 8 OOD count. |
| 5 | Prostate XFedGA β unknown (deck 0.2, script default 0.5) | Prostate XFedGA column of slide 11; slide 6 (β = 0.2 prostate); the prostate parts of slides 12 and 18. |
| 6 | Grad-CAM layer failed a known-answer test (old half-resolution layer, all-pixel target) | Explanation Consistency and Pointing-Game rows of slides 10 and 11; slide 12 explanation conclusions; and, via q_i and the EC fitness term, every XFedGA number on slides 10 and 11. |
| 7 | PROMISE12 client cases are Decathlon server scans: 12 duplicated cases, 8 in the server sets (`reports/audit/leakage_note.md`) | Prostate: every number on slide 11; prostate rows of slides 8–9; the prostate parts of slides 12 and 18. Breast numbers are not affected (`reports/audit/breast_duplicates.md`). |
| 8 | Unequal fitness data: XFedGA scored fitness on the first 40 images of the unshuffled fitness loader, FedGA (baseline) on the whole fitness set (audit #37, D-5) | Every FedGA-vs-XFedGA comparison: all rows of slides 10 and 11 and the conclusions on slides 12 and 18 (the two methods differed in fitness data as well as in method). |
| 9 | "Gradient-aware perturbation" claimed but not implemented (audit §2.9) | None: the phrase is not in the Review-2 deck, so no Review-2 number depends on it. It affects the novelty statement only. |
