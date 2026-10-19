# Breast v1 headline: EC (IoU@0.5) and EC_mass on every test image

Declared evaluation of the fixed, validation-selected checkpoints (no selection). Source: `analysis/ec_mass_headline.py` → `results/xai_ec/breast_headline_ec.json`.

| Method | EC (IoU@0.5): mean ± s.d. over seeds | EC_mass: mean ± s.d. over seeds |
|---|---|---|
| FedAvg | 0.218 ± 0.083 | 0.440 ± 0.148 |
| FedGA (baseline) | 0.212 ± 0.081 | 0.446 ± 0.160 |
| XFedGA | 0.223 ± 0.084 | 0.435 ± 0.158 |

| Comparison | Metric | Patients | Mean diff. | 95% CI (cluster bootstrap) | Outcome |
|---|---|---|---|---|---|
| XFedGA vs FedGA (baseline) | ec_iou | 638 | +0.0110 | [+0.0085, +0.0137] | WIN |
| XFedGA vs FedGA (baseline) | ec_mass | 638 | -0.0105 | [-0.0126, -0.0084] | LOSE |
| XFedGA vs FedAvg | ec_iou | 638 | +0.0049 | [+0.0024, +0.0075] | WIN |
| XFedGA vs FedAvg | ec_mass | 638 | -0.0044 | [-0.0063, -0.0026] | LOSE |
| FedGA (baseline) vs FedAvg | ec_iou | 638 | -0.0061 | [-0.0089, -0.0034] | LOSE |
| FedGA (baseline) vs FedAvg | ec_mass | 638 | +0.0061 | [+0.0040, +0.0082] | WIN |

Check: max |recomputed IoU EC − stored test `ec`| over all runs = 2.13e-04.
