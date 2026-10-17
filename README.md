# XFedGA

XFedGA is explanation-aware federated learning with server-side genetic-algorithm (GA) search, for medical image segmentation (M.Tech project, Amrita Vishwa Vidyapeetham).
- Each client sends its model update and a client-side Grad-CAM saliency score **q_i**.
- The server aggregates with weights ∝ (n_i/N)·max(q_i, eps)^beta, then runs a GA whose fitness combines DSC, HD95 and explanation consistency.
- The best candidate becomes the global model.

How the pieces fit: [ARCHITECTURE.md](ARCHITECTURE.md). Live demo script: [docs/DEMO.md](docs/DEMO.md). Research spec: [docs/XFEDGA_SPEC.md](docs/XFEDGA_SPEC.md). Current plan: [reports/audit/plan.md](reports/audit/plan.md).

> Status (17 Oct 2026): Session 3a — all method and scenario switches implemented and smoke-tested; experiment queue in `analysis/run_queue.py`. Results produced before 13 Oct are superseded: their checkpoints were selected on the test set (see [results/legacy/README.md](results/legacy/README.md)).

## Tasks and data
| Task | Clients | Server data (fitness / validation / test) |
|---|---|---|
| Breast ultrasound | BUSI, BUS-UCLM | BUS-BRA, split by patient 15 / 25 / 60 % |
| Prostate MRI | PROMISE12, split 50/50 by case into 2 clients | Decathlon Task05, cases 8 / 8 / 16 |

Expected locations (relative to the repo root): `raw data/BUSI`, `raw data/BUS-UCLM`, `raw data/BUS-BRA`, `raw data/PROMISE12`, `raw data/Task05_Prostate/Task05_Prostate`. All datasets are public and de-identified; see their licences (Decathlon: CC-BY-SA 4.0).

**Preprocessing:**
- every image and mask is resized to 384×384 before training and before every metric;
- breast images are min-max normalised;
- prostate slices are resampled to 0.5 mm in-plane, clipped to the 1st/99th percentile and min-max normalised;
- only slices or images with a non-empty mask are used.

Preprocessed tensors are cached under `cache/`, built automatically on first use and rebuilt whenever the preprocessing code or raw files change.

## Setup (Windows, PowerShell)
```powershell
# Python 3.10 environment with the pinned packages
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
The development environment is `D:\conda_envs\fedga`. The commands below use `$py = "D:\conda_envs\fedga\python.exe"`; with a venv, use `python` instead.

## Run
```powershell
$py = "D:\conda_envs\fedga\python.exe"

# tests (add -m "not slow" to skip the GPU tests)
& $py -m pytest

# one federated run: --task breast|prostate, --seed N, --method one of
#   fedavg | fedga (FedGA baseline = ablation A) | xfedga (= ablation E)
#   abl_B | abl_C | abl_D | abl_F | random_search | fedprox | centralized | local_only
# scenario switches (configs, scenario:): partition=acquisition (prostate), noisy_qi_client,
#   corrupted_label_client, dp.enabled
& $py experiments/run_federated.py --task prostate --method xfedga --seed 1

# smoke run: 1 round, tiny GA, no test evaluation
& $py experiments/run_federated.py --task breast --method xfedga --seed 1 --no_test --set train.rounds=1 ga.population=4 ga.generations=2

# any hyperparameter can be overridden with dotted keys
& $py experiments/run_federated.py --task breast --method xfedga --seed 1 --set aggregation.beta=1.0
```
Each run writes `results/runs/<run_id>/`:
- `config_resolved.yaml`
- `round_log.csv`: validation DSC / HD95 / EC / pointing game, q_i and alpha_i per client, GA best/mean fitness per generation, and timings
- `test_per_image.csv`: the selected checkpoint on the test set, with patient/case ids
- `run_meta.json`: run id, seed, config hash, git commit, selected round
- the validation-selected checkpoint `<task>_<method>_seed<k>_valselected.pt`

The split case/patient ids are logged in `results/splits/<task>_dataseed<seed>.json` (identical for every run; each run checks it) and, with the fixed subsamples, in the run folder (`split_log.json`).

## Experiment queue
```powershell
# sequential, resumable; skips finished runs; logs GPU state and flags throttling
& $py analysis/run_queue.py queues/beta_screen.yaml
# run from a frozen worktree so later edits cannot change running jobs:
git worktree add --detach ..þdga_frozen\<sha> <sha>
& $py ..þdga_frozen\<sha>nalysis
un_queue.py ..þdga_frozen\<sha>\queues\<queue>.yaml --repo .
```

## Repository layout
`configs/ data/ models/ fedga/ xai/ experiments/ fl_dist/ (Session 3) analysis/ tests/ reports/ results/`; see [ARCHITECTURE.md](ARCHITECTURE.md).

The pre-audit training scripts (`experiments/train_fedga.py`, `train_fedga_prostate.py`, `train_baseline_fedavg.py`), the root-level analysis scripts and `scripts/` were **removed on 15 Oct 2026** (superseded by `experiments/run_federated.py` and `analysis/`). They remain in git history at tag `legacy-scripts-final`; their outputs are in `results/legacy/` (superseded, OOD-selected).
