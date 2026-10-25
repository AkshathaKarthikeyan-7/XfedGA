# Base FedGA paper vs our `mode=base`

**Source.** `D:\Amrita\Project\Literature review\FedGA_Genetic_Algorithm-Guided_Federated_Learning_for_Medical_Image_Segmentation_with_Non-IID_Features.pdf`, 14 pages, "IEEE Journal of Biomedical and Health Informatics", DOI 10.1109/JBHI.2026.3676321 (author's accepted version).
- Text was extracted with pypdf 6.20.1 on 13 Oct 2026. Page numbers below are PDF page indices.
- Nothing here is quoted from memory. The authors are cited only as they appear on page 1: Faisal Ahmed, Rodrigo Moreno, David Sánchez, Zouhair Haddi, Josep Domingo-Ferrer.

## 1. Setup comparison
| Item | Paper (page) | Our `mode=base` (file:line) | Match? |
|---|---|---|---|
| Breast clients / server | 6 sites from BUS-BRA (split by scanner), BUSI, QAMEBI. OOD experiment: site F (BUS-BRA U-Systems) is the server, sites A–E are clients (p.6, p.8 Table II, p.10) | 2 clients (BUSI, BUS-UCLM); server = all of BUS-BRA | **no** (different federation) |
| Prostate clients / server | 6 sites: NCI-ISBI13 (A, B), I2CVB (C), PROMISE12 split by source (D BIDMC, E HK, F UCL). Site F is the server (p.6, p.8 Table I, p.10) | 2 synthetic clients (PROMISE12 random 50/50); server = Decathlon Task05 | **no** |
| Rounds | 20 (p.8) | 15 (`train_fedga.py:182`) | **no** |
| Local epochs | 10 (p.8) | 10 | yes |
| Optimizer | Adam, β1 = 0.9, β2 = 0.99, lr 1e-3 (p.8) | same (`train_fedga.py:49`) | yes |
| Batch size | 128 (p.8) | 8 | **no** (6 GB GPU) |
| Local early stopping | yes: client 80/20 train/val, stop if DSC does not improve for 3 epochs, keep best local epoch (p.7–8) | none; the 20% client val split is discarded (`train_fedga.py:207`) | **no** |
| Loss | equal-weighted Dice + CE (p.8) | `DiceCELoss(lambda_dice=0.5, lambda_ce=0.5)` (`fedga/losses.py`) | yes |
| Model | 2-D Residual U-Net [16, 32, 64, 128, 256], strides 2, 2 residual units, for breast; **Attention U-Net for prostate** (p.7) | Residual U-Net for both tasks | breast yes, prostate **no** |
| Preprocessing, breast | grayscale, per-image min-max to [0,1], resize to 384×384 (p.6) | same | yes |
| Preprocessing, prostate | FOV centred on the prostate, 384×384 axial, z-score intensity (p.6) | resample to 0.5 mm, then 1st/99th-percentile clip + min-max, resize to 384×384, no FOV crop | **no** |
| Initial population | averaged model + Gaussian neighbours ĝ = g + η, η ~ N(0, σ²); **10** models; original included so the "solution will not perform worse" (p.4, p.8) | w_agg + (P−1) noisy copies, **P = 6** | method yes, size **no** |
| σ (noise) | **not stated** in the paper (searched for σ value / sigma / 0.01) | 0.01 | unknown |
| Generations | **20** (p.8) | **5** | **no** |
| Fitness | mean per-image DSC on server fitness data, Eq. 5 (p.5) | `evaluate_dsc` mean per-image DSC on the full fitness set | yes |
| Selection | elitism: sort by DSC, keep best N/2 (p.5; Alg. 1 line 11, p.6) | same (`ga_optimizer.py:105`) | yes |
| Crossover | uniform, per weight r < 0.5 → P1, else P2, Eq. 6 (p.5); applied with probability Pc = 0.5 (p.8, Alg. 1) | same (`ga_optimizer.py:36-42, 110`) | yes |
| Mutation | per-weight Gaussian with probability α = Pm = 0.1, Eq. 7 (p.5–6, p.8) | same (`ga_optimizer.py:45-52`), fixed σ in base mode | yes |
| Final pick | arg max fitness over the population (Alg. 1 line 16, p.6) | same (`ga_optimizer.py:119-121`) | yes |
| Server split | server data split into 2 mutually exclusive parts, one for GA fitness and one for evaluation (p.7); "50% of patient data" for fitness (p.9) | 50/50, but **per image** for BUS-BRA (D-2); per case for Decathlon | breast **no** |
| Round selection | "performance of the best global model across all federated training rounds on the server data" (p.10) | old runs: best round chosen on the OOD half (`train_fedga.py:267`). New protocol: a separate validation set | **deliberate difference** (see below) |
| Statistics | per-image Shapiro–Wilk (p = 0.05) → paired t-test or Wilcoxon (p.10) | same (`compare_significance.py`) | yes (spec §6 replaces it with case-level cluster bootstrap) |
| GA ablation | Pc ∈ {0.4, 0.5, 0.6} × Pm ∈ {0.05, 0.1, 0.2}, breast; Pc = 0.5 / Pm = 0.1 best (p.13, Fig. 9) | not run | – |
| Hardware | NVIDIA A10 24 GB (p.8) | RTX 4050 Laptop 6 GB | – |

**Round/checkpoint selection: a deliberate difference.** The paper reports "the best global model across all federated training rounds on the server data" (p.10). Its server data has only two parts, fitness and evaluation (p.7), so the round is selected on the same data used to report OOD performance. We deliberately differ: the server data is split three ways (fitness / validation / test). Rounds and checkpoints are selected on the **validation set only**, and the test set is used once per method. This applies to every method we run, including FedGA (baseline) and the optional paper-faithful reference run. Our numbers are therefore expected to be lower than best-round-on-test numbers, and the report must say so.

**Naming (binding for all reports and slides).**
- Our `mode=base` is named **"FedGA (baseline)"** in all tables, figures and slides. It is never called a reproduction, and no "reduced budget" wording is used in labels.
- The difference from the paper is stated **once**:
  - in a setup table in the report's methods section, listing paper vs ours for population, generations, rounds, batch size, local early stopping, federation and prostate model, with the page references from §1 below;
  - in one line on the experimental-setup slide;
  - with this sentence: *"FedGA and XFedGA are run with identical settings, which differ from the original paper as listed."*

**Conclusion.** Our `mode=base` implements the GA *operators* of Algorithm 1 faithfully: fitness, elitism, uniform crossover, sparse Gaussian mutation, and an initial population that includes w_agg. It does **not** reproduce the paper's *setup*:
- population 6 vs 10; 5 vs 20 generations; 15 vs 20 rounds; batch 8 vs 128;
- no local early stopping; a different federation and server site; Residual U-Net for prostate; different prostate preprocessing.

Its docstring says it "reproduces the original FedGA exactly" (`fedga/ga_optimizer.py:5`). In Session 2 (after approval), reword it to "FedGA (baseline): implements the GA operators of Algorithm 1 of the base paper; settings differ from the paper as listed in the methods setup table". The chosen budget (P = 6, G = 5) is justified by the measured compute in plan.md.

An optional **paper-faithful reference run** (P = 10, G = 20, 20 rounds, batch 8, 1 seed per task, selected on validation) is costed in plan.md. Batch 128 is checked for fit on 6 GB by a memory test (see `results/timing/`), not assumed. Even this run differs from the paper in federation, server data, model (prostate) and early stopping, so it is labelled "FedGA (paper GA settings)", never a reproduction.

### Setup table for the report/methods (draft)
| Setting | Paper (page) | Ours: FedGA (baseline) and XFedGA |
|---|---|---|
| Population | 10 (p.5, p.8) | 6 |
| Generations | 20 (p.8) | 5 |
| Rounds | 20 (p.8) | 15 |
| Batch size | 128 (p.8) | 8 |
| Local early stopping | yes, client 80/20, patience 3 (p.7–8) | no |
| Federation | 5 clients + 1 server site per task (p.6, p.8 Tables I–II, p.10) | 2 clients + external server dataset per task |
| Prostate model | Attention U-Net (p.7) | Residual U-Net |
| Round selection | best round on server data (p.10) | separate validation set |

*FedGA and XFedGA are run with identical settings, which differ from the original paper as listed.*

## 2. Paper-reported results (for citation only; not comparable to our numbers)
**Table V (p.11).** OOD generalisation, server = site F, mean ± std. DSC in %, HD95 in mm.

| Method | Breast DSC | Breast HD95 | Prostate DSC | Prostate HD95 |
|---|---|---|---|---|
| FedAvg | 81.77 ± 2.42 | 59.14 ± 2.67 | 64.64 ± 2.97 | 83.23 ± 3.24 |
| FedProx | 82.62 ± 1.48 | 60.62 ± 1.98 | 58.47 ± 2.90 | 97.28 ± 2.73 |
| SCAFFOLD | 72.06 ± 2.31 | 83.22 ± 3.27 | 24.44 ± 2.44 | 150.40 ± 2.40 |
| FedBN | 65.25 ± 2.78 | 110.34 ± 2.63 | 20.95 ± 3.85 | 75.52 ± 3.58 |
| FedCLAM | 78.83 ± 1.91 | 62.37 ± 1.52 | 57.98 ± 1.98 | 89.21 ± 1.93 |
| FedGA | 84.52 ± 1.58 | 54.16 ± 1.92 | 66.29 ± 1.52 | 72.13 ± 1.43 |

(The p-values against FedGA are in the same table, p.11.)

**Table X (p.13).** Average runtime per round over 20 rounds (breast): FedAvg 904.18 s, FedGA 1040.18 s, overhead **+15.04%**.

**Table XI (p.13).** "Optimal runtime" (per-round runtime × rounds to the best round): FedAvg 17179.42 s, FedGA 13522.34 s (−21.29%). This depends on best-round selection on server data (p.12: FedGA best at round 13, FedAvg at round 19).

## 3. Points relevant to our claims
1. **GA overhead is much larger here than the paper reports.** The paper reports +15% over FedAvg with P = 10, G = 20 (p.13). Our measured GA share of a round is far higher (see `results/timing/timing_summary.csv`). Likely reasons: a small local workload (2 clients, small data, batch 8 on a laptop GPU) and repeated elite evaluation (audit D-9). Do not cite the paper's 15% for our setup.
2. **Expect our absolute DSC to be lower than Table V's.** The setups differ (2 vs 5 clients, other server data, budget), so Table V numbers must not be placed next to ours as a comparison. The spec (§9B) asks only that the trend (FedGA ≥ FedAvg) is checked under our setup.
3. **Paper Table II vs BUS-BRA on disk.** Patient counts match the scanners on disk: Toshiba 72, GE Logiq 5 490, GE Logiq 7 488, U-Systems 14. Image counts do **not**: the paper lists 902 / 1064 / 809 / 25, while `bus_data.csv` gives 139 / 809 / 902 / 25. Note this before citing Table II; do not attempt to explain it.
4. **The paper's prostate site F (UCL) is a PROMISE12 source** (Table I, p.8), and it lists PROMISE12 sources by institute (BIDMC, HK, UCL). The per-case source assignment is **not** on our disk (audit §3), so our acquisition-profile groups cannot be relabelled as those institutes without a documented case→site mapping.
