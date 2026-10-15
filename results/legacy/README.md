# Legacy results (superseded; kept for the record)

Everything in this folder predates the Session-1 audit (13 Oct 2026) and is **superseded**. Do not cite these numbers as results.

**Why superseded** (`reports/audit/audit.md`):
- Every `*_BEST_OODselected.pt` was chosen as the round with the highest DSC **on the OOD test set**, verified by exact re-evaluation in `results/audit/verify_checkpoints.json`. This is selection on the test set.
- All runs are single and unseeded. Statistics were per image, not per case.
- Breast data bugs: malignant BUS-UCLM lesions were silently dropped (red masks; audit D-1), and the BUS-BRA fitness/OOD split leaked 382 patients across the halves (audit D-2).
- Prostate XFedGA: the beta used is unknown (the Review-2 deck says 0.2; the script default is 0.5).

| Folder | Origin | Contents |
|---|---|---|
| `breast/` | repo root, runs of 13–17 Aug 2026 | FedAvg, FedGA (baseline) and XFedGA: final-round `.pt`, `*_BEST_OODselected.pt`, `round_history_*.csv` |
| `prostate/` | repo root, runs of 26 Sep – 2 Oct 2026 | FedGA (baseline) and XFedGA, same file types |
| `nested_xfedga_2026-08-13/` | the deleted nested `xfedga/` folder | older breast runs (q_i from 20 samples), CSVs with DSC only |
| `figures/` | repo root | Grad-CAM comparison figures made from the OOD-selected checkpoints |

The Review-1 and Review-2 deck numbers came from these checkpoints and are marked superseded (decision Q10, 15 Oct 2026). `.pt` and `.png` files are git-ignored; only the CSVs and this README are tracked.
