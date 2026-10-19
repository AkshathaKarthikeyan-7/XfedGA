# Legacy re-evaluation (DIAGNOSTIC, not a claim)

Old breast FedGA (baseline) vs old XFedGA, **final-round** checkpoints (not the test-selected ones), on the current BUS-BRA test patients that were not in the old fitness half: 329 images, 206 patients (432 of 638 test patients excluded). Current Grad-CAM. Source: `analysis/legacy_reeval.py` → `results/diagnostics/legacy_reeval.json`.

| Metric | Old FedGA | Old XFedGA | Δ (patient-level) [95% CI] | Corrected outcome | Per-image test (Review-2 style) | Review-2-style verdict | Review-2 claim (slide 10) | Claim survives? |
|---|---|---|---|---|---|---|---|---|
| DSC | 0.563 | 0.534 | -0.0270 [-0.0482, -0.0057] | LOSE | Wilcoxon, p = 0.0045 | Worse | Tied (p = 0.30) | no |
| HD95 (normalised) | 0.341 | 0.383 | +0.0396 [+0.0236, +0.0556] | LOSE | Wilcoxon, p = 3.6e-10 | Worse | Better (p < 0.0001) | no |
| EC (IoU@0.5) | 0.138 | 0.041 | -0.0957 [-0.1042, -0.0871] | LOSE | paired t-test, p = 1.3e-78 | Worse | Better (p = 0.0066) | no |
| EC_mass | 0.253 | 0.561 | +0.3021 [+0.2713, +0.3330] | WIN | Wilcoxon, p = 1.6e-47 | Better | — (not on slide 10) | — |
| Pointing game | 0.739 | 0.690 | -0.0364 [-0.0898, +0.0170] | TIE | exact McNemar, p = 0.081 | Tied | Better (p < 0.0001) | no |

**Which Review-2 breast conclusions survive** (rule fixed in the script before any result: the corrected patient-level outcome must match the slide-10 claim): none survive; do not survive: DSC, HD95 (normalised), EC (IoU@0.5), Pointing game.

This removes only the test-set checkpoint selection (audit item 1), the fitness/test patient overlap (item 4) and the per-image statistics (item 2, by showing both). It cannot remove what is baked into the old models: BUS-UCLM malignant masks dropped in training (item 3), single unseeded runs (item 2), the old Grad-CAM layer in q_i and the EC fitness (item 6) and unequal fitness data (item 8). The per-image column shows how much the statistics method alone changes the verdict.
