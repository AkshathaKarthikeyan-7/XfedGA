# Session 1 — end-of-session report (audit and plan), 13–15 Oct 2026

## 1. Files added, changed or deleted
**Existing repo files: none changed, none deleted.** All new files are under `reports/audit/` and `results/timing/`. Git `HEAD` = `766c8cc`, and the new files are untracked and not committed.

| File | Reason |
|---|---|
| `reports/audit/audit.md` | Inventory, item-by-item verification of spec §0, discrepancies D-1…D-12 |
| `reports/audit/fedga_paper_comparison.md` | Paper vs our setup with page refs (pypdf); "FedGA (baseline)" naming; methods setup-table draft; p.10 round-selection note |
| `reports/audit/plan.md` | Costed Tier 1 / Tier 2 plan, splits, `fl_dist/` design, β design, cuts, hypotheses, decisions |
| `reports/audit/plan_costs.py` → `plan_costs.json`, `plan_costs.md` | Cost model and fill-order schedule, computed from the measured timings |
| `reports/audit/verify_checkpoints.py` → `verify_checkpoints.json` | Shows every `*_BEST.pt` was selected on the OOD set |
| `reports/audit/summarize_existing.py` → `existing_results.json` | Best / final / last-5 summary of the old round histories |
| `reports/audit/case_level_variance.py` → `case_level_variance.json` | Per-case SD of paired ΔDSC (old fitness halves only), for CI planning |
| `reports/audit/session1_status.md` | Pause/resume status |
| `reports/audit/session1_report.md` | This report |
| `reports/audit/env_freeze_after_pypdf_pytest.txt` | Added **by you** (pip freeze after the pytest + pypdf install) |
| `results/timing/timed_run.py` | Timing harness: runs the existing scripts unmodified, monkey-patched timers, outputs isolated in run folders |
| `results/timing/summarize_timing.py` → `timing_summary.{csv,json}` | Per-run timing summary |
| `results/timing/batch_memcheck.py` → `batch_memcheck.json` | One forward/backward per batch size |
| `results/timing/runs/<run>/` (8 folders) | Per-run `timing_events.jsonl`, `stdout.log`, `config_abs_paths.yaml`, plus the scripts' own `*.pt` / `round_history_*.csv` outputs (timing artefacts only, not results) |

Note: CLAUDE.md wants analysis scripts in `analysis/`. That folder could not be created this session, so the scripts above should move there in Session 2.

## 2. Test suite
No test suite exists, before or after: **0 passed / 0 failed**. pytest 9.1.1 is now installed (by you, 14 Oct); the minimal suite is part of Session 2.

## 3. Runs completed
All runs are timing measurements with the **existing, unmodified code**, which is unseeded. Git commit at run time: `766c8cc`. The config hash is SHA-256 (first 12 hex characters) of the run's `config_abs_paths.yaml`.

| Run id | Task | Method | Seed | Rounds | Config hash | Commit |
|---|---|---|---|---|---|---|
| smoke_prostate_xfedga | prostate | XFedGA (P = 4, G = 2, 1 epoch) | unseeded | 1 | 8af6812fb652 | 766c8cc |
| smoke_breast_xfedga | breast | XFedGA (P = 4, G = 2, 1 epoch) | unseeded | 1 | 851b1079773a | 766c8cc |
| full_prostate_base | prostate | FedGA (baseline) | unseeded | 15 / 15 | 8af6812fb652 | 766c8cc |
| full_prostate_xfedga | prostate | XFedGA | unseeded | 8 / 15 (stopped by you) | 8af6812fb652 | 766c8cc |
| full_breast_xfedga | breast | XFedGA | unseeded | 15 / 15 | 851b1079773a | 766c8cc |
| full_breast_base | breast | FedGA (baseline) | unseeded | 3 / 15 (auto-stop: projection 1.64 h > 1 h) | 851b1079773a | 766c8cc |
| share2_prostate_base_p1 / _p2 | prostate | FedGA (baseline), 2 runs in parallel | unseeded | 2 / 2 each | 8af6812fb652 | 766c8cc |

Also: the checkpoint re-evaluation, the case-level variance estimate and the batch memory check (evaluation only, no training). Also a throwaway Flower toy federation (CPU, scratch folder outside the repo).

## 4. Compute used vs the costed plan
- **This session: ≈ 2.7 GPU-h of wall clock.** That is 2.75 h summed over the run folders, counting the two parallel sharing runs once, plus a few minutes of evaluation-only scripts.
- No costed plan existed before this session. The plan budgets ≈ 3 GPU-h for the Session 2–3 build and **104 GPU-h** (4 h per weekday, 10 per weekend day, 23 Oct – 8 Nov) for experiments.

## 5. Three-line summaries
> **Tested:** Wall clock of one 15-round run per task, FedGA (baseline) and XFedGA, current code and config (prostate XFedGA 8 rounds, breast FedGA 3 rounds, both extrapolated).
> **Result:** Per round: prostate 131 s (FedGA) / 210 s (XFedGA); breast 394 s (FedGA, full 938-image fitness set) / 223 s (XFedGA). 36 fitness evaluations per round; GA is 43–68% of the round; peak 0.71–0.74 GB reserved (`timing_summary.csv`).
> **Implies:** One XFedGA run costs ≈ 0.8–1.0 GPU-h. Tier 1 = 79.7 GPU-h raw (103.6 with contingency) against a 104-h budget, so the fill order in plan §8 decides what is cut.

> **Tested:** 2 prostate FedGA runs in parallel on the one GPU, 2 rounds each, vs rounds 1–2 of the single run.
> **Result:** 248 s/round each vs 118 s single, a throughput factor of 0.95; local training ~3× slower; 712 MiB per process (`timing_summary.csv`).
> **Implies:** No gain from parallel runs: the loop is CPU/data-loading-bound. Session 2 Step 0 (tensor cache + workers) is the cheapest likely speed-up; recost after the re-measure.

> **Tested:** One forward/backward/Adam step at 384×384 for batch sizes 8–128.
> **Result:** Reserved memory 0.67 / 1.31 / 2.56 / 4.78 / 7.21 / 9.55 GB; batch ≥ 96 exceeds the 6 GB of dedicated VRAM (`batch_memcheck.json`).
> **Implies:** The paper's batch 128 cannot run here; the optional "FedGA (paper GA settings)" run uses batch 8.

> **Tested:** Re-evaluating every root checkpoint on the OOD split, matched against `round_history_*.csv`.
> **Result:** Every `*_BEST.pt` reproduces the max-OOD-DSC round exactly (|Δ| ≤ 3e-8): breast FedAvg r1, FedGA r2, XFedGA r2; prostate FedGA r8, XFedGA r2 (`verify_checkpoints.json`).
> **Implies:** All earlier headline numbers are OOD-selected and superseded. On the last-5 mean, XFedGA trails FedGA (baseline) on both tasks in those single runs.

> **Tested:** Per-case SD of the paired ΔDSC between the two existing prostate (and breast) models, on the old fitness halves only.
> **Result:** Prostate SD 0.070, so a 95% CI of ±0.038 with 16 test cases (±0.059 with 8); breast SD 0.137 per patient, so ±0.011 with 600 patients (`case_level_variance.json`).
> **Implies:** Decathlon split 8 / 8 / 16 maximises test precision. On prostate only |ΔDSC| ≳ 0.04 is detectable, so TIE is the likely, pre-registered outcome.

## 6. Open questions for you
These are the decisions Q1–Q10 in `plan.md` §10; each has a stated default.
1. Q1: server splits (Decathlon 8 / 8 / 16; BUS-BRA 15 / 25 / 60% of patients).
2. Q2: FedProx in Tier 1 at fill position 10.
3. Q3: client validation splits unused.
4. Q4: HD95 in pixels / 384.
5. Q5: move legacy outputs and delete the nested `xfedga/`.
6. Q6: parity margin ±0.02.
7. Q7: elite cache, conditional on the determinism test.
8. Q8: exact Review 3 date.
9. Q9: drop "gradient-aware perturbation".
10. Q10: mark the Review-1/2 numbers superseded.

**Approval requested** for `plan.md`, including Session-2 Step 0 (data-loading speed-up + re-measure + recost). No code will change until you approve.
