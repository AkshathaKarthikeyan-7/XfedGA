# Diagnostics v1: headline runs (18 Oct 2026)

**Scope:** validation data, training logs and client / server-fitness / server-validation masks only. **No test data was read.** No training: item 4 uses inference with the saved validation-selected checkpoints on validation slices.
- **Source:** `analysis/diagnostics_v1.py` → `results/diagnostics/v1.json`.
- **Runs:** the 30 headline runs (frozen code `b556334`, new CAM, β = 0.5).

## 1. GA overfitting: fitness gain vs validation gain
| Run group | Rounds | GA winner ≠ w_agg | Fitness gain of winner over w_agg (fitness subset): mean / median / p90 |
|---|---|---|---|
| prostate FedGA (baseline) | 75 | 97.3% | 0.0223 / 0.0194 / 0.0432 (DSC fitness) |
| prostate XFedGA | 75 | 98.7% | 0.0147 / 0.0130 / 0.0265 (multi-objective) |
| breast FedGA (baseline) | 75 | 98.7% | 0.0129 / 0.0107 / 0.0218 (DSC fitness) |
| breast XFedGA | 75 | 98.7% | 0.0084 / 0.0068 / 0.0155 (multi-objective) |

**The validation DSC of w_agg is not logged.** Only the GA winner is scored on validation each round. So the requested correlation between fitness gain and validation gain, and how often the winner is worse than w_agg on validation, **cannot be computed from the existing logs**.
- **What the logs do show:** the GA almost always replaces w_agg (97–99% of rounds), but by small fitness margins on the 40-image fitness subset. A gain of that size is comparable to the subset's sampling noise. So the "improvement" may be a selection effect on 40 images, not a real gain.
- **To answer it:** log w_agg's validation metrics every round (one extra validation pass, about 5 s prostate / 17 s breast per round). That's a code change for future runs, not made here.

## 2. Early peak and drift (validation DSC per round, mean over 5 seeds)
| Run group | Peak round per seed | Selected round per seed | Peak val DSC | Round-15 val DSC | Peak → round 15 |
|---|---|---|---|---|---|
| prostate FedAvg | 8, 13, 4, 6, 1 | 8, 1, 5, 7, 1 | 0.440 | 0.354 | −0.086 |
| prostate FedGA | 5, 5, 5, 1, 1 | 4, 6, 6, 1, 6 | 0.432 | 0.351 | −0.081 |
| prostate XFedGA | 5, 5, 7, 2, 3 | 6, 2, 10, 1, 2 | 0.461 | 0.345 | −0.115 |
| breast FedAvg | 2, 2, 2, 3, 3 | 1, 3, 3, 3, 3 | 0.666 | 0.590 | −0.076 |
| breast FedGA | 2, 4, 2, 3, 2 | 1, 3, 3, 3, 1 | 0.675 | 0.597 | −0.078 |
| breast XFedGA | 2, 2, 2, 2, 2 | 1, 3, 3, 3, 3 | 0.679 | 0.594 | −0.085 |

- **Every method peaks early** (breast at round 2–3; prostate mostly at rounds 1–8) and then **loses 0.08–0.12 validation DSC by round 15**. The external server domain gets worse with more federated training.
- **Client training loss is not logged**, so "clients still improving their own loss after the peak" cannot be checked directly.
- **Proxy:** the mean client update norm ‖w_i − w_global‖ falls from about 26 (round 1) to 16–18 by round 3 and **stays flat until round 15**, on both tasks and for all methods. Clients keep moving the model by the same amount every round after the validation peak. That is consistent with continued fitting to the client domains, i.e. **drift away from the server domain**, but it does not prove it.
- **Implication (validation-based, for the next runs, not a test comparison):** 15 rounds × 10 local epochs is more training than the server domain benefits from. Fewer local epochs or rounds, or logging client loss, would be the follow-up.

## 3. Prostate preprocessing: how much of the slice is gland
| Set | Slices | Gland fraction of the 384×384 slice: mean / median |
|---|---|---|
| PROMISE12 client 1 (train) | 379 | 3.1% / 2.9% |
| PROMISE12 client 2 (train) | 399 | 3.7% / 3.3% |
| Decathlon fitness set | 117 | 2.7% / 2.8% |
| Decathlon validation set | 117 | 4.4% / 3.5% |

- **The gland is a small object: about 3–4% of each slice.**
- **Ours:** resample to 0.5 mm in-plane → 1st/99th-percentile clip → min-max to [0,1] → resize the whole slice to 384×384, with no crop.
- **The base FedGA paper** (`reports/audit/fedga_paper_comparison.md`, from the PDF p.6): "a consistent field of view centred on the prostate region", resized to 384×384, intensities **z-score** normalised.
- **What this suggests:** an FOV crop would make the gland a much larger share of the image. That is the most obvious candidate for our low prostate DSC (~0.40 validation). It is a hypothesis to test on validation, not a result.

## 4. Prostate validation error analysis: false-positive blobs and largest-component post-processing
| Method | Slices × seeds | Val DSC raw | Val DSC after largest component | Δ | LCC helps / hurts (share of slices) | Components per slice | Predicted foreground not touching GT |
|---|---|---|---|---|---|---|---|
| FedAvg | 585 | 0.408 | 0.400 | −0.009 | 39% / 36% | 8.4 | 31% |
| FedGA (baseline) | 585 | 0.394 | 0.376 | −0.018 | 24% / 42% | 5.1 | 31% |
| XFedGA | 585 | 0.416 | 0.397 | −0.018 | 34% / 39% | 7.7 | 32% |

- **False positives outside the gland are substantial:** about 31–32% of predicted foreground lies in components that do not touch the ground truth, with 5–8 components per slice.
- **But keeping only the largest connected component does not help** (Δ = −0.009 to −0.018). The predictions are fragmented, so the largest component is often not the gland, or the gland is split and LCC cuts true parts.
- A size threshold or a 3-D (per-case) largest component might behave differently; that is untested.
- **Conclusion:** a substantial part of the DSC loss is detached false-positive blobs, but naive per-slice LCC is the wrong fix.
