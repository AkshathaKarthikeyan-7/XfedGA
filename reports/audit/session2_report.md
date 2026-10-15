# Session 2 — end-of-session report (hygiene, splits, seeding, core refactor), 15 Oct 2026

Branch `session-2`, 11 commits from `619cafb` (Session 1) to `ffe9c48`. Not merged and not pushed.

## 1. Files added, changed or deleted
| Change | Files | Reason |
|---|---|---|
| **Step 0: data-loading speed-up** | `data/cache.py` (new), `data/loading.py` (new), `data/breast_dataset.py`, `data/prostate_dataset.py`, `data/server_split.py`, 3 legacy `experiments/train_*.py` (loader options, cache) | Preprocessed 384×384 tensors cached per dataset under `cache/` (git-ignored, ≈1.8 GB), keyed by a SHA-256 of the preprocessing code, parameters and raw-file fingerprints. Cached output equals on-the-fly output exactly. `num_workers` is configurable in YAML, **default 0** (see §6) |
| **D-1** | `data/breast_dataset.py` | BUS-UCLM masks are read in colour and binarised on the max channel. The 90 malignant lesions are kept (264 lesion images, was 174) |
| **D-2 + subsamples** | `data/splits.py` (new) | BUS-BRA 15 / 25 / 60 % of **patients** (stratified device × pathology) → 159 / 267 / 638 patients, 278 / 470 / 1,127 images. Decathlon **8 / 8 / 16 cases** (117 / 117 / 241 slices). BUS-UCLM client split by patient prefix. Fixed 40-image fitness subsample and fixed 40-image q_i subsamples |
| **Core refactor (algorithm in `fedga/`, `xai/`)** | `fedga/client.py`, `fedga/evaluation.py` (new); `fedga/fedavg.py` (explicit `aggregation_weights`); `fedga/ga_optimizer.py` (tau, per-generation history, elite cache; docstring no longer claims an exact reproduction); `xai/explainability.py` (`unet_cam_layer`) | Framework-agnostic functions, for reuse by `fl_dist/` in Session 3 |
| **Runner + config** | `experiments/run_federated.py`, `experiments/tasks.py`, `experiments/config.py` (new); `configs/*.yaml` | One seeded loop for FedAvg / FedGA (baseline) / XFedGA. Every hyperparameter is in YAML and validated (beta and fitness weights exposed for both tasks; HD95 normaliser = image side, images resized to 384 before metrics — Q4). Validation every round; checkpoint = best centred 3-round mean validation DSC; **test evaluated once**, per image with patient/case ids. Elite cache enabled (Q7) |
| **Tests** | `tests/` (8 files), `pytest.ini` | 73 tests (see §2) |
| **Cleanup (Q5)** | `results/legacy/` (new, README); nested `xfedga/` **deleted** (17 tracked files + untracked copies) | Legacy checkpoints renamed `*_BEST_OODselected.pt`; legacy CSVs and figures moved; the nested 13-Aug outputs kept |
| **Scripts → `analysis/`** | `verify_checkpoints`, `summarize_existing`, `case_level_variance`, `plan_costs`, `timed_run`, `summarize_timing`, `batch_memcheck` | CLAUDE.md layout; outputs now in `results/audit/` and `results/timing/` (doc references updated) |
| **Docs** | `README.md` (rewritten), `ARCHITECTURE.md` (new, one data-flow diagram), `requirements.txt` (pinned to the env, incl. pytest 9.1.1 and pypdf 6.20.1), `.gitignore` (`cache/`, `results/smoke/`) | Session-2 deliverables |
| **Plan** | `reports/audit/plan.md` §0 (new: project plan to 7 Nov), §0-old marked superseded, Q1–Q10 answers recorded | Your re-plan instruction |
| **Results** | `results/splits/{breast,prostate}_dataseed42.json`, `results/timing/runs/s0_*`, `results/timing/runner/` (+ `runner_throttled/`), `results/audit/plan_costs.*` | Split ids; timings; costs |
| Unchanged, legacy | `experiments/train_fedga.py`, `train_fedga_prostate.py`, `train_baseline_fedavg.py`, root analysis scripts | Still working; **not deleted without your approval**; README marks them legacy |

## 2. Test suite
- **Before: 0 tests** (no suite existed).
- **After: 73 passed, 0 failed** (`python -m pytest`, 3 min 56 s, including 6 slow GPU tests; `-m "not slow"` gives 67 in about 25 s).
- **Coverage:**
  - aggregation weights sum to 1 (β grid; β = 0 equals FedAvg);
  - the elitism guarantee (the winner is never worse than w_agg, and w_agg is kept when it is the optimum);
  - GA evaluation counts, and that the elite cache does not change the search;
  - split disjointness at patient/case level, the proportions, and that the split log does not depend on subsample settings;
  - **seed determinism** (same seed → bit-identical model and metrics; different seed → different model);
  - fitness repeatability (the Q7 precondition);
  - **FedGA and XFedGA receive the same fitness subsample** on both tasks;
  - cache equals on-the-fly preprocessing on all 5 datasets, and works with Windows `num_workers > 0`;
  - D-1 counts;
  - HD95 conventions vs the legacy implementation;
  - config validation.

## 3. Runs completed (timing and smoke only; no results runs)
| Run id | Task | Method | Seed | Rounds | Config hash | Commit |
|---|---|---|---|---|---|---|
| s0_{prostate,breast}_{base,xfedga} | both | legacy scripts after Step 0 | unseeded | 3 | per-folder `config_abs_paths.yaml` | e81e138 (Step 0) |
| s0_share2_prostate_base_p1/p2 | prostate | legacy FedGA ×2 in parallel | unseeded | 2 | same | e81e138 |
| prostate_xfedga_seed1_32efc170_smoke | prostate | XFedGA (P = 4, G = 2) | 1 | 1 | 32efc1708fb4 | 6c45422-dirty |
| breast_xfedga_seed1_a1bfed70_smoke | breast | XFedGA (P = 4, G = 2) | 1 | 1 | a1bfed708201 | 6c45422-dirty |
| {prostate,breast}_{fedavg,fedga,xfedga}_seed1_*_timing3 (`results/timing/runner/`) | both | all 3 | 1 | 3 | fa4926a2ca6d, 6db06e44c894, e9c9524a86da, 77502d976021, 6a994a0d7c5d, a91dc6f9b082 | 4779ca8-dirty (only the uncommitted cost script differed) |
| same, `runner_throttled/` | both | 5 runs, 1 stopped | 1 | 3 | same | same; **power-capped GPU, not used** |

## 4. Compute used vs the costed plan
- **This session ≈ 1.9 GPU-h:** 1.47 h of measured runs, plus ≈ 0.4 h of tests, profiling and cache builds.
- **Plan for Session 2:** ≈ 0.9 GPU-h (re-measure) + smokes.
- **Why it was higher:** a power-capped first timing pass, which had to be repeated, and a full 6-run runner timing for the recost.
- **New plan:** 73.0 GPU-h of 192 available (18 Oct – 5 Nov, 8 h per weekday night and 16 h per weekend day). The queue finishes 25 Oct; no cuts are needed (`plan.md` §0.3).

## 5. Three-line summaries
> **Tested:** Step 0 (tensor cache), legacy scripts, 3 rounds per task and method, vs the 13–14 Oct timings.
> **Result:** Breast local training 113 → 38 s/round; round time prostate FedGA 131 → 70 s, XFedGA 210 → 127 s; breast FedGA 394 → 142 s, XFedGA 223 → 133 s (`results/timing/timing_summary.csv`, extrapolated).
> **Implies:** The loop is now GPU-bound (util ~92%); the cache was the main win. `num_workers > 0` gave no further gain (measured).

> **Tested:** 2 runs in parallel on one GPU after Step 0 (prostate FedGA, 2 rounds).
> **Result:** 165 s/round each vs 70 s single, a throughput factor ≈ 0.85 (`timing_summary.csv`).
> **Implies:** Run the queue sequentially; parallel runs slow everything down.

> **Tested:** New runner, 3 rounds, FedAvg / FedGA / XFedGA × 2 tasks, full GPU clocks (median 2,730 MHz, no throttle flags).
> **Result:** Per round: prostate 55 / 58 / 109 s, breast 65 / 68 / 123 s. XFedGA's extra cost is the GA's multi-objective fitness (52–56 s vs 3 s for DSC-only). 15-round runs take 0.23–0.53 GPU-h (`results/timing/runner/`, `results/audit/plan_costs.json`).
> **Implies:** The whole project queue is 73 GPU-h. FedGA (baseline) costs about the same as FedAvg once it uses the shared 40-image subsample.

> **Tested:** Whether FedGA (baseline) and XFedGA score the same fitness subsample (spy test + run split logs, both tasks).
> **Result:** Identical 40 image ids on both tasks; 21 evaluations per round each (`tests/test_fitness_subsample.py`; split logs in `results/timing/runner/*`).
> **Implies:** Weakness 6 (subsample confound) is fixed. The earlier slower breast FedGA came from the legacy scripts (full 938-image set).

> **Tested:** Seed determinism and fitness repeatability (prostate, tiny runs, GPU).
> **Result:** Same seed → bit-identical weights, metrics, q_i, alpha_i and GA history; repeated fitness identical (`tests/test_determinism.py`). Deterministic kernels cost ≈ 24% per training epoch (profiled).
> **Implies:** Q7 condition met; the elite cache is enabled (36 → 21 evaluations per round).

## 6. Open questions for you
1. **Approve `plan.md` §0** (project plan to 7 Nov), including the flagged assumptions:
   - 3 seeds for the β screen, fitness-weight grid, heterogeneous partition and robustness;
   - FedProx at 1.1× local time;
   - `fl_dist` at 1.15×;
   - XAI metrics on a fixed 300-image breast subsample.
2. **`num_workers` deviation.** You asked for `num_workers > 0`. Measured with the cache, workers give no steady-epoch gain and 8–30 s start-up, so the YAML default is **0** (workers are supported and tested on Windows). Keep 0?
3. **Start the headline queue early?** It could start on the 16–17 Oct nights (+24 GPU-h buffer) before Session 3a, protected by a regression test that keeps the 3 existing methods unchanged. Approve?
4. **Spare ≈ 119 GPU-h.** Use it for 5-seed ablations, the full β grid {0, 0.2, 0.25, 0.5, 1.0}, or a 5-seed heterogeneous partition? Not added without approval.
5. **Legacy scripts.** Delete `experiments/train_*.py` and the root analysis scripts (superseded by the runner and `analysis/`)? They still work.
6. **Git.** Merge `session-2` into `main`? Push to `origin`, which the open-source-repository TEA needs? Nothing has been pushed.
7. **Keep the power mode on Best performance** for every queue night. A power-capped GPU doubled run times today.
8. **Limitation to note in the paper:** BUSI has no patient ids, so its client train/validation split is per image. The client validation split is held out and unused (Q3), so this does not affect any reported metric.
