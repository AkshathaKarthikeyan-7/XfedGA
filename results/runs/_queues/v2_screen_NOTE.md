# v2_screen queue note (18 Oct 2026)

- 17:50:24: `results/runs/STOP` was created to skip the two prostate seed-2 runs, which are VOID (PROMISE12 -> Decathlon leakage, `reports/audit/leakage_note.md`).
- Run 5 (prostate_fedga_seed2) had already finished at 17:29:01, and run 6 (prostate_xfedga_seed2) had been running since 17:29:01. STOP acts only between runs, so it would have stopped the queue after run 6, i.e. before the two breast runs, saving no GPU time.
- STOP was therefore removed at once (before run 6 ended). The ORIGINAL queue continues unchanged with the breast seed-2 runs (7, 8). No remainder queue was used, so the breast run IDs and config hashes are those of the original queue by construction.
- All prostate v2-screen runs (1, 2, 5, 6) are SUPERSEDED / void (plan.md section 0.L).
