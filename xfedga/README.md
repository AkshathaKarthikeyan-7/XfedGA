# XFedGA — Explainable Genetic-Algorithm-Guided Federated Learning
### M.Tech Final Year Project

Upgrades the FedGA paper (Ahmed et al., IEEE JBHI 2026) with:
1. A multi-objective, explanation-aware fitness function for the GA
2. Adaptive (annealed) mutation instead of fixed-σ Gaussian noise
3. Explanation-weighted client aggregation (replaces plain FedAvg weighting)
4. Grad-CAM / attention-based explainability integrated into both training and evaluation

---

## 1. Client / dataset mapping (breast lesion segmentation)

| Client | Dataset | Status |
|---|---|---|
| Client 1 | BUSI (Breast Ultrasound Images Dataset) | ✅ downloaded |
| Client 2 | BUS-BRA | ✅ downloaded |
| Client 3 | BUS-UCLM | ⏳ planned |

## 2. Client / dataset mapping (prostate segmentation — phase 2)

| Client | Dataset | Status |
|---|---|---|
| Client 4 | PROMISE12 | ⏳ registering this weekend |
| Client 5 | I2CVB | ⏳ downloading next week |

Each dataset = one simulated hospital/client, consistent with the original FedGA paper's
cross-silo setup (data heterogeneity comes from the different scanners/protocols
naturally present across these public sources — no need to artificially partition).

## 3. Suggested phased plan

**Phase 0 — Data pipeline (start here, this week)**
- Preprocessing: grayscale, min-max normalize to [0,1], resize to 384×384 (matches
  base paper, keeps results comparable)
- Per-client train/val split (80/20)
- Sanity-check: compute FID between client pairs to confirm heterogeneity (reuse the
  base paper's Fig. 2/3 method) — good early figure for your report

**Phase 1 — Baseline reproduction**
- Implement plain FedAvg + 2D Residual U-Net on your 2 (then 3) breast clients
- Reproduce a mini version of the base paper's Table III/V numbers as a sanity check
  before adding anything new

**Phase 2 — GA module (core of FedGA)**
- Implement Algorithm 1 as-is first (single-objective DSC fitness) — this is your
  "baseline FedGA" ablation for the paper

**Phase 3 — Your contributions (this is the paper)**
- Multi-objective fitness (DSC + HD95 + explanation consistency)
- Annealed mutation
- Explanation-weighted aggregation
- Grad-CAM integration + faithfulness metrics (pointing game, insertion/deletion AUC)

**Phase 4 — Prostate task (once PROMISE12/I2CVB land)**
- Repeat with Attention U-Net; use attention-gate maps as a second, free explainability
  channel — nice comparison point (learned attention vs. post-hoc Grad-CAM)

**Phase 5 — Writing**
- **Conference paper** (shorter, faster venue): breast-only results, core method
  (multi-objective fitness + explanation-weighted aggregation), positioned as "adding
  explainability to metaheuristic FL"
- **Journal paper** (extended): add prostate task, full ablations, faithfulness
  metrics, out-of-domain generalization tests — mirrors the depth of the base FedGA
  JBHI paper

---

## 4. Repo layout
```
xfedga/
  data/           # dataset-specific loaders + preprocessing
  models/         # Residual U-Net, Attention U-Net
  fedga/          # FedAvg, GA module (population, fitness, crossover, mutation)
  xai/            # Grad-CAM, attention-map extraction, faithfulness metrics
  configs/        # YAML configs per experiment
  experiments/    # training/eval entry points
  scripts/        # one-off utilities (FID heatmap, dataset stats)
```

## 5. Environment
See `configs/environment.md`. Core deps: `torch`, `monai`, `scikit-learn`,
`opencv-python`, `pytorch-grad-cam`, `pyyaml`.
