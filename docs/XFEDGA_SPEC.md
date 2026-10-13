# XFedGA — master research spec

Save this as `docs/XFEDGA_SPEC.md` in the repo. `CLAUDE.md` holds the standing rules (environment, non-negotiables, vocabulary, end-of-session report). This file says what to build and how to judge it. Work happens in separate sessions (section 11). Do only what the current session's prompt asks.

## ROLE
You are a senior AI scientist in medical imaging and federated learning. The project is XFedGA: explainability-aware federated learning with server-side genetic-algorithm (GA) optimization for medical image segmentation (breast ultrasound and prostate MRI). It will be defended before an academic panel (Amrita Vishwa Vidyapeetham, M.Tech AI). Every claim must be backed by an experiment run in this repo.

## TIMELINE AND BUDGET
- Phase 1 ends at Review 3, November 2026, about 4 weeks from 13 Oct 2026. Phase 2 runs Dec 2026 – Apr 2027.
- I have an internship, so compute and my time are limited.
- Before planning any experiment, measure the wall-clock time of one real 15-round run per task. Cost every grid in hours before proposing it.
- **Tier 1 (by Review 3) covers priorities 0–3 in section 10 only.** Everything else is Tier 2 unless the measured timings show Tier 1 has room.

## 0. CURRENT STATE (notes to verify, not guarantees)
- Files (verified by the first git commit, 13 Oct 2026): `experiments/train_fedga.py` (breast), `experiments/train_fedga_prostate.py`, `experiments/train_baseline_fedavg.py`, `fedga/fedavg.py`, `fedga/ga_optimizer.py`, `fedga/losses.py`, `xai/explainability.py`, `models/residual_unet.py`, `data/` (breast_dataset, prostate_dataset, server_split), `configs/` (breast_clients.yaml, prostate_clients.yaml, environment.md), `scripts/` (Grad-CAM checks), root-level analysis scripts (compare_significance*, gradcam_comparison*, debug_busbra, diagnose_class_balance, inspect_model, test_gradcam), and `round_history_*.csv` files.
- The custom simulation is single-process, single-GPU, with a sequential client loop. MONAI provides the model and HausdorffDistanceMetric.
- **`fl_dist/` does NOT exist.** A search of D:\ and the user folder found no such folder. There is also **no `tests/` folder and no test suite.** Treat every earlier mention of "fl_dist, 35 tests, ablations A–E, FedProx, server fine-tune already built" as false until you find code proving otherwise.
- **Nested `xfedga/` folder:** this appears to be an older, breast-only duplicate of the root code. Diff it against the root and report what differs. Propose removing it (after I approve), keeping anything unique.
- **To build** (the tier in section 10 decides when):
  - a test suite
  - ablations B–F in the custom loop
  - the FedProx and server fine-tune baselines, if they are absent
  - the GA-free random-search control
  - NSD
  - the three-way fitness/validation/test split
  - the independent explanation metrics (section 6)
  - the robustness and DP experiments (section 7)
  - the multi-process Flower federation (section 7)
- Breast: the clients are BUS-UCLM and BUSI. The server data is BUS-BRA.
- Prostate: PROMISE12 is split randomly 50/50 by case into 2 synthetic near-IID clients. The server data is Decathlon Task05.
- Defaults: 15 rounds, 10 local epochs, Adam lr 1e-3, batch 8, population 6, generations 5, pc 0.5, pm 0.1, sigma0 0.01.
- Aggregation (`fedga/fedavg.py`): alpha_i ∝ (n_i/N)·max(q_i, 1e-3)^beta, normalized.
  - beta defaults to 0.5. It was dampened from 1.0 because q_i is computed from about 20 images and swings between about 0.04 and 0.20 from round to round.
  - The prostate script has `--beta`. The breast script does not.
  - There is no post-GA blend.
- q_i = `compute_client_saliency_score`, computed client-side on the client's own training loader, max 20 samples, using `explanation_consistency(cam, mask, threshold=0.5)`, with a fallback of 0.5.
- GA:
  - Population: one unchanged copy of w_agg plus (P−1) Gaussian-noise copies at sigma0. The initial population is never annealed.
  - Each generation: rank by fitness, keep the top half (elitism), create offspring by uniform crossover (pc) and sparse Gaussian mutation (pm, sigma = sigma0·exp(−g/10) in xfedga mode).
  - About 36 fitness evaluations per round.
- Fitness: 0.5·DSC − 0.3·HD95_norm + 0.2·EC.
  - The weights are hard-coded, not exposed in config, and have no tuning evidence.
  - HD95 is normalized by 384 and clipped to [0,1].
  - XFedGA scores a 40-image subsample with Grad-CAM per image. Base FedGA scores the whole fitness set.
  - Base mode = DSC-only fitness, fixed sigma, plain FedAvg.
- **Claimed contributions to verify.** Earlier slides describe three upgrades: multi-objective fitness, Grad-CAM XAI, and *gradient-aware perturbation*. Check whether any gradient-aware perturbation exists in `fedga/ga_optimizer.py` or elsewhere. If it does not, report this. It must then be removed from the slides, the Review 2 deck and the novelty statement, or implemented and ablated in Tier 2.
- **Existing results, now superseded.** These came from checkpoints selected on the OOD set (`*_BEST.pt`) and were tested at slice level. They must be re-run. In the final report, explain how and why the re-run numbers differ.
  - Prostate, 233 OOD slices / 16 cases, base FedGA vs XFedGA: DSC 0.544 vs 0.496; HD95_norm 0.243 vs 0.215; EC 0.120 vs 0.197; pointing game 60.9% vs 57.5% (p=0.34).
  - Breast: find the current breast numbers in the repo's existing results/logs and record them in the audit, with the file each came from. Earlier summary: XFedGA better on explainability with no accuracy cost.
- Known weaknesses:
  1. The best round is selected on the OOD test set. This applies to prostate; check breast too.
  2. Annealing barely decays in 5 generations (0.01 → about 0.0067).
  3. EC is both optimized and reported, so it is circular.
  4. The prostate partition is near-IID, so q_i weighting has little to separate.
  5. Base FedGA and XFedGA differ in several things at once, so an ablation is needed.
  6. Base FedGA scores the full fitness set while XFedGA scores 40 images. This is a confound.
  7. Client loaders are shuffled, so q_i uses different images each round.
  8. The fitness weights were hand-set.
  9. Decathlon Task05 has only 32 labelled cases.
  10. The metric name is not verified. Check whether `explanation_consistency` is IoU, and name it exactly as the code computes it.
- Other files:
  - Panel deck: search the repo and `D:\Amrita\Project\` for `XFedGA_Panel_Review*.pptx` and report the path. If it is not found, ask me.
  - Datasets: all are inside `D:\new repo\xfedga` in separate folders. Locate each dataset (BUSI, BUS-UCLM, BUS-BRA, PROMISE12, Decathlon Task05) and record its path in the audit.
  - Slide 6 must read: local training → client-side Grad-CAM score → explanation-weighted aggregation (beta = dampening exponent) → GA search → best-candidate selection (no blend) → broadcast.

## 1. RESEARCH QUESTIONS
State each hypothesis and its null hypothesis in writing before running any experiment.
- **RQ1** Does explanation-weighted aggregation plus multi-objective GA selection improve OOD segmentation (DSC, HD95, NSD) over FedAvg, FedProx and base FedGA?
- **RQ2** Does it improve explanation quality, measured with metrics the method does not optimize?
- **RQ3** Which component contributes the gain: weighted aggregation, multi-objective fitness, annealing, or the GA itself?
- **RQ4** Is it robust across seeds, client heterogeneity and hyperparameters?
- **RQ5** Does the multi-process Flower federation behave the same as the single-process simulation?

## 2. DATA AND SPLITS
- Split at patient/case level, never at slice level. Log the case IDs of every split to disk. Report n_i in cases and slices.
- Divide the server data into three disjoint sets:
  - **fitness set:** GA scoring only.
  - **validation set:** all round, checkpoint and hyperparameter selection.
  - **test set:** used once per method, at the end.
- **Decathlon allocation.** 32 cases cannot fill three sets generously. Propose an allocation with a justification, for example a smaller fitness set and balanced validation and test sets. Report the CI width this allows. Pre-registered expectation: on prostate, "comparable" accuracy is the likely outcome, and a TIE is a valid result.
- Prostate partition:
  - Check the PROMISE12 documentation for site metadata. If there is none, say so.
  - Run both the random 50/50 split and a heterogeneous split (intensity/acquisition-profile skew, or Dirichlet over case characteristics).
- Subsamples:
  - Use one fixed, deterministic fitness subsample, the same for all candidates and for both base FedGA and XFedGA.
  - Use one fixed q_i subsample per client.
  - Run one check on the full fitness set to show the subsample does not change conclusions.
- Document licenses, the ethics statement (public de-identified data), preprocessing, empty-mask filtering and its effect on class balance.

## 3. MODEL AND TRAINING
- Residual U-Net (MONAI), Dice+CE loss, settings as in section 0, all from YAML config.
- Include a centralized upper bound and a local-only lower bound for each task.
- Seed everything deterministically. The original script is unseeded; fix that first.

## 4. METHOD (clean specification)
Round t:
1. The server sends the global model to each client.
2. Each client trains locally, then computes q_i on its own data (Grad-CAM vs its own masks). No images or masks leave the client. Define q_i mathematically, exactly as the code computes it.
3. Each client sends w_i, n_i and q_i.
4. The server computes alpha_i ∝ (n_i/N)·max(q_i, eps)^beta, normalizes it, and forms w_agg.
5. GA:
   - The population is w_agg (unchanged) plus noisy copies.
   - Offspring come from uniform crossover and Gaussian mutation, with elitist selection.
   - Fitness F = w1·DSC − w2·HD95_norm + w3·EC on the fitness set.
6. The best candidate becomes the global model. Unit-test that w_agg is never dropped, so that fitness(winner) ≥ fitness(w_agg) on the fitness set. This guarantee holds on the fitness set only.

Fix or justify each of these with an experiment:
- **Annealing:** sweep tau ∈ {1, 2, 5, 10}, and also anneal the initial-population noise. Otherwise drop the "annealing" claim.
- **HD95 normalizer:** 384 must match the real image size or diagonal from config.
- **beta:** expose it in CLI/config for both tasks. Sweep {0, 0.25, 0.5, 1.0} and report the round-to-round variance of q_i. Claim that "beta=1.0 is unstable" only if that run shows it.
- **Fitness weights:** expose them in config. Tune per 5B, and make the final choice on the validation set only.

## 5. BASELINES AND ABLATIONS
- **Baselines:** centralized; local-only; FedAvg; FedProx; server fine-tune; base FedGA; and FedAvg + random-perturbation search (GA-free, equal number of fitness evaluations).
- **Ablations**, all else equal:
  - A: FedAvg + GA(DSC) = base FedGA
  - B: weighted aggregation only, no GA
  - C: FedAvg + GA(multi-objective)
  - D: FedAvg + GA(DSC) + annealing
  - E: full XFedGA
  - F: full XFedGA with random q_i

  Build all ablations as mode/config switches in the custom loop, so that no algorithm logic is duplicated. A probably equals the existing `mode=base`; verify this.
- **Sensitivity:** beta, population {4, 6, 10}, generations {3, 5, 10}, sigma0, tau. Report wall clock and number of fitness evaluations for each.

## 5B. FITNESS-WEIGHT TUNING
- **Step 0 (cheap):** re-rank one saved GA candidate pool offline under many (w1, w2, w3) settings and report how often the winner changes.
  - If it rarely changes, test only the corners, 0.5/0.3/0.2, and a few other points.
  - If it changes often, run a simplex grid with step 0.1 on breast only, with 3 seeds and 8 rounds. Then run the top 3 combinations at 15 rounds with 5 seeds.
- Constraints: w1 + w2 + w3 = 1, all w ≥ 0.
- Select on the validation set only: mean DSC and HD95, with ties broken by EC and seed stability.
- Report:
  - a ternary/heatmap of validation DSC
  - mean ± std per combination
  - a paired comparison against 0.5/0.3/0.2
  - transfer from breast to prostate
  - the Pareto front
  - the number of GA runs spent on tuning
  - a note that the weighted sum is a scalarization choice
- If results are insensitive to the weights, report that as a finding.

## 6. EVALUATION PROTOCOL
- **Statistics:**
  - The primary test is paired at case level: average over seeds within each case, then compare methods.
  - Use a cluster-bootstrap 95% CI over cases, plus effect sizes. Seeds quantify training variance and are reported separately.
  - A Wilcoxon test over 5 seeds cannot be significant (minimum p = 0.0625).
  - Claim an improvement only if the CI excludes 0. Otherwise report "comparable".
- **Seeds:** 5 for the headline comparison (priority 2). 3 for ablations and sensitivity, unless the timings allow 5.
- **Segmentation metrics:** DSC, HD95, NSD, ASSD, precision/recall; per client and OOD; per-case failure analysis (empty predictions, tiny lesions); final-round and last-5-round mean ± std.
- **Explanation metrics:** use only metrics the method does not optimize: pointing game, deletion/insertion AUC, a second CAM method (Grad-CAM++ or Score-CAM), and the Adebayo parameter-randomization check. EC alone cannot prove better explanations. IoU with ground truth measures plausibility, not faithfulness.
- **Optional:** ECE; Dice by lesion size; per-client DSC spread.

## 7. FEDERATED SYSTEM REALISM
- **Level 1, custom simulation:** source of the main results. Justification: the research question is the aggregation algorithm, and a single process gives reproducibility and fair comparison.
- **Levels 2–3, `fl_dist/` (to build, Tier 2):**
  - Build thin Flower wrappers that import the same core functions as the custom loop.
  - Use the Flower 1.7.0 API (`start_server` / `start_numpy_client`).
  - Run the server and clients as separate OS processes over loopback gRPC, with a message audit proving that only weights, n_i and q_i are exchanged.
  - Parity check against the custom loop: 3 seeds, breast task, compared statistically.
- **Measure:** bytes per round per client, rounds to reach a target DSC, GA overhead (extra evaluations, server time). Localhost latency is not meaningful.
- **Untested, and must be stated as such:** network faults, stragglers, heterogeneous hardware, cross-institution deployment.
- **Privacy:** make no guarantee claim. Discuss weight-leakage risk. Optionally run clipping + Gaussian noise and report the DSC cost. Secure aggregation is out of scope.
- **Robustness:** run one client that always reports q_i = 1, and one client with corrupted labels. Report the effect on DSC and on the aggregation weights.

## 8. ENGINEERING AND REPO HYGIENE
- Client and server core functions stay framework-agnostic, shared by the custom loop and `fl_dist/`.
- **Tests:** create `tests/` (pytest, if it is already installed in the env; otherwise ask me) covering:
  - aggregation weights sum to 1
  - fitness weights sum to 1 and are ≥ 0
  - the elitism guarantee
  - HD95 edge cases
  - case-level split disjointness
  - determinism under a fixed seed
  - custom vs Flower aggregation equality on a toy example

  The whole suite runs with one command.
- **No magic numbers:** every hyperparameter (beta, fitness weights, sigma0, tau, population, generations, subsample sizes, HD95 normalizer, seeds) lives in YAML under `configs/` and is loaded in one place.
- **Code hygiene:** no `*_v2.py`, `*_final.py`, `*_new.py` or `*_old.py` files; no dead code, commented-out blocks or debug prints. Use logging. Docstrings state assumptions, for example "q_i is computed client-side".
- **Layout:** `configs/ data/ models/ fedga/ xai/ experiments/ fl_dist/ analysis/ tests/ reports/ results/`. Include:
  - `.gitignore` covering `*.pt`, raw data, caches and logs
  - pinned requirements matching the existing env
  - a README with exact PowerShell commands
  - `ARCHITECTURE.md` with one data-flow diagram
- **Logs and results:**
  - Per-round CSV logs: DSC, HD95, NSD, EC, pointing game, q_i per client, and GA best/mean fitness per generation.
  - Plot fitness-set vs validation-set improvement to detect GA overfitting.
  - Record run id, seed, config hash and git commit for every run. Add a git tag per reported result.
- **Checkpoints:** name them like `breast_xfedga_seed1_valselected.pt`. Mark the old `*_BEST.pt` files as OOD-selected; do not delete them silently.
- **Tooling:** add ruff/black only if they install without touching pinned versions.
- **Final step:** run a clean-clone check. Clone to a new folder, follow only the README, and confirm the 1-round smoke run and the test suite both pass.

## 9. DELIVERABLES
- Clean code and a README, including the commands for `fl_dist` and for fitness-weight tuning.
- Figures generated by scripts from `results/`:
  - DSC vs round
  - ablation bars
  - beta sensitivity
  - fitness-weight re-ranking/heatmap
  - Pareto front
  - custom vs Flower comparison
  - overlays (image, GT, prediction, Grad-CAM) for the best, median and worst cases and for failure cases
- Report sections: motivation, related work (FedAvg, FedProx, FedGA, explainable FL), method, experiments, limitations, ethics, future work.
  - Limitations must include: single-machine federation, 2 clients per task, synthetic partition, small Decathlon sets, weighted-sum scalarization, EC circularity, no formal privacy guarantee, not a clinical device.
- About 15 slides consistent with the code. Every number traces to a results file. Any tool named on a slide must actually have been run.
- A panel Q&A document with at least 40 questions, answers and evidence pointers. Topics include: why GA, fitness weights, beta, why simulation, what the multi-process run proves, why 2 clients, test leakage, statistics, faithfulness vs plausibility, compute, privacy, clinical relevance, malicious client.

## 9B. "BETTER THAN EXISTING WORK" — PRE-REGISTERED
Existing work means base FedGA (mode=base) and FedAvg/FedProx, run under identical conditions. Claim nothing against methods not run here.

**Fair comparison**
- Reproduce base FedGA first and compare it with the paper's setup and trend, citing the paper's table from the PDF.
- Every method uses identical splits, model, epochs, rounds, seeds and server sets.
- Compare at matched compute: equal wall clock and equal number of fitness evaluations, plus the random-search control.
- Give baselines the same validation tuning budget, including sigma/pm/pc for base FedGA.
- Evaluate the test set once per method.

**Four axes, each scored WIN / TIE / LOSE per task**
1. **Accuracy:** a case-level paired difference with a 95% CI excluding 0. Otherwise TIE.
2. **Explanation:** independent metrics only.
3. **Stability:** seed std, last-5-round std, per-client gap, heterogeneous partition, noisy-q_i client.
4. **Cost:** extra compute and bytes per round, reported as a trade-off.

Report claims must match the outcome table exactly.

**Example outcome-table row:**
| Task | Accuracy | Explanation | Stability | Cost |
|---|---|---|---|---|
| Prostate | TIE (ΔDSC −0.01, CI [−0.05, 0.03]) | WIN (pointing game +4.1 pts, CI [0.8, 7.2]) | TIE | +38% wall clock/round |
(The numbers above are illustrative only. Real rows come from `analysis/`.)

**Strengthening and fallback**
- Changes come only from validation evidence. Candidates: EMA-smoothed q_i, the fixed fitness subsample, adaptive beta. Keep a change only if it wins on validation across seeds. Document the rejected ideas.
- Write a novelty statement listing what is new vs base FedGA and which parts the ablation supports. Drop the unsupported parts.
- Fallback positioning: comparable accuracy with better independent explanation metrics at X% extra compute, or a well-analysed negative result.

## 10. WORKING RULES
- Inspect first and verify every item in section 0. List the discrepancies before changing anything.
- Run a smoke test (1 round, small population) before every full run.
- If an expected result does not appear, report it and propose a follow-up on the validation set.
- For each design decision, state the alternative considered and why it was rejected.
- **Priority order:**
  - Tier 1: (0) repo hygiene, config cleanup, removal of the nested duplicate and a minimal test suite; (1) three-way split, fixed subsamples and seeding; (2) FedAvg vs base FedGA vs XFedGA; (3) ablations including F and the random-search control.
  - Tier 2: (4) fitness-weight Step 0 and beta sensitivity; (5) heterogeneous prostate partition; (6) custom vs fl_dist parity; (7) NSD and independent explanation metrics; (8) robustness and DP.
  - If the timings allow, pull (7) into Tier 1. Without independent explanation metrics, RQ2 cannot be answered.

**3-line experiment summary format (use exactly this):**
> **Tested:** E vs A on prostate, random split, 5 seeds, val-selected checkpoints.
> **Result:** ΔDSC −0.012 (CI [−0.048, 0.021], 10 test cases); ΔHD95_norm −0.018 (CI [−0.041, 0.004]).
> **Implies:** comparable accuracy; HD95 trend favours E but is not significant; next is the heterogeneous partition.
(Illustrative numbers.)

## 11. SESSION PLAN
Each session ends with the end-of-session report from `CLAUDE.md`.
1. **Audit and plan.** Verify section 0, measure timings, produce the costed Tier 1/Tier 2 plan. Then stop for approval.
2. **Hygiene, splits and seeding** (priorities 0–1).
3. **Core comparison** (priority 2).
4. **Ablations** (priority 3).
5. **Tier 2 items**, one per session.
6. **Write-up:** report, slides, Q&A.
