# XFedGA

Explanation-aware federated learning with a server-side genetic algorithm (GA), for medical image segmentation.
M.Tech project, Amrita Vishwa Vidyapeetham.

- **FedGA (baseline)** (the base paper's algorithm): the server averages the client models (w_agg) and runs a GA around w_agg. Its fitness is DSC on a server fitness set.
- **XFedGA**: each client also sends a Grad-CAM saliency score q_i. The server weights the clients by (n_i/N)·max(q_i, eps)^β and uses the fitness F = w1·DSC − w2·HD95_norm + w3·EC.
- **XFedGA+** (a declared post-v1 variant): the server scores each client model itself (s_i), and the GA searches over client mixing weights.

Every experiment was pre-registered before its runs (`docs/protocol/plan.md`). Checkpoints are selected on validation only, and the test set is evaluated once per run.

## Folder structure
| Path | Contents |
|---|---|
| `fedga/` | algorithm: aggregation, GA (`ga_optimizer.py`), fitness and metrics (`evaluation.py`), client update, losses |
| `xai/` | Grad-CAM / Grad-CAM++, explanation consistency (EC, EC_mass), pointing game, deletion / insertion, randomisation checks |
| `experiments/` | the single-process federated loop (`run_federated.py`), task builders (`tasks.py`), config handling |
| `fl_dist/` | Flower 1.7.0 version: multi-process federation on a single machine over loopback gRPC |
| `data/` | dataset loaders, preprocessing cache, splits, scenarios (label corruption) |
| `configs/` | one YAML per task (`<task>_clients.yaml`: `breast`, `prostate_ms`, `prostate_ms_runmc`; `prostate` is superseded) |
| `queues/` | the exact run lists of every experiment (YAML) |
| `analysis/` | scripts that produce every reported table and figure; `analysis/archive/` = planning and queue generators |
| `results/` | all run outputs, tables, figures and audits (each script writes here) |
| `docs/protocol/` | pre-registrations (`plan.md`), audits, the leakage note, the Review-2 comparison, panel responses |
| `docs/` | research spec (`XFEDGA_SPEC.md`), Flower demo script (`DEMO.md`), environment freeze |
| `reports/` | generated diagnostic and review reports |
| `tests/` | pytest suite (fast CPU tests; `-m slow` = GPU tests) |

## Data
Expected under `raw data/` (all public, de-identified):
- **Breast:** `BUSI`, `BUS-UCLM` (clients) and `BUS-BRA` (server; split by patient 15 / 25 / 60 % into fitness / validation / test).
- **Prostate:** `MultiSiteProstate/{BIDMC,BMC,HK,I2CVB,RUNMC,UCL}`.
  - `prostate_ms`: clients BIDMC, HK, I2CVB, RUNMC, UCL; server BMC.
  - `prostate_ms_runmc`: clients BIDMC, BMC, HK, I2CVB, UCL; server RUNMC.
  - Server cases are split 8 / 8 / 14 into fitness / validation / test.
- The earlier PROMISE12 → Decathlon setting is **superseded**: 12 PROMISE12 client cases are Decathlon server scans (`docs/protocol/leakage_note.md`).

## Setup (Windows, PowerShell; Python 3.10)
```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # full freeze: docs/environment/requirements_freeze.txt
$py = "python"                           # the commands below use $py
& $py -m pytest -m "not slow"            # fast tests (CPU); drop -m to run the GPU tests too
```

## Reproduce the results
**Runs.** Every experiment is a queue. `analysis/run_queue.py` runs it sequentially, skips finished runs and logs the GPU state. Run it from a frozen worktree at the commit named in the pre-registration:
```powershell
git worktree add --detach ..\xfedga_frozen\<sha> <sha>
& $py ..\xfedga_frozen\<sha>\analysis\run_queue.py ..\xfedga_frozen\<sha>\queues\<queue>.yaml --repo .
# a single run:
& $py experiments/run_federated.py --task breast --method xfedga --seed 1 [--no_test] [--set key=value ...]
```

**Tables and figures.** These run on the CPU and read `results/runs/`:

| Result | Queue(s) | Command |
|---|---|---|
| Final headline (PRIMARY: XFedGA vs FedGA; secondary: XFedGA+ vs FedGA) | `headline.yaml` (breast), `final_headline.yaml` | `& $py analysis/make_tables.py --group main,final --comparisons final --tasks breast,prostate_ms --out results/tables/final` |
| XAI metrics on checkpoints | as above | `& $py analysis/xai_metrics.py --runs "results/runs/*_final"` |
| Protocol / XFedGA+ / acceptance decisions (validation) | `v2_screen.yaml`, `screen_19oct.yaml`, `acceptance_screen.yaml` | `analysis/v2_adoption.py`, `analysis/xplus_adoption.py`, `analysis/acceptance_adoption.py` |
| a. Dose-response | `post_a_dose_response.yaml` | `& $py analysis/dose_response.py` |
| b–f. Flower parity, paper GA settings, H-ROB2, ablations, FedProx + bounds | `post_b_*` … `post_f_*` | `& $py analysis/post_items_analysis.py` |
| g. Independent site (RUNMC) | `post_g_site_runmc.yaml` | `& $py analysis/g_site_runmc.py` |
| g2. Breast replication (seeds 6–10) | `post_g2_breast_replication.yaml` | `& $py analysis/g2_replication.py` |
| h / i. Fitness-weight and β sensitivity (validation) | `post_h_fw_screen.yaml`, `post_i_beta_screen.yaml` | `& $py analysis/sensitivity_screens.py` |
| Root-cause diagnostics | (all screens) | `& $py analysis/root_cause.py` |
| Final figures | (all) | `& $py analysis/final_figures.py` |
| Leakage and duplicate audits | — | `analysis/promise12_decathlon_overlap.py`, `analysis/multisite_inventory.py`, `analysis/breast_duplicate_audit.py` |

## Flower demo
The demo uses breast with 2 clients; the script is in `docs/DEMO.md`.
```powershell
& $py fl_dist/launch.py --task breast --method xfedga --seed 1 --tag demo --set train.rounds=2 train.local_epochs=2
```
One server process and one OS process per client communicate over loopback gRPC. Each run writes `message_audit.json` (exactly weights, n_i, q_i and the client index per message).

## Main results
Test DSC, paired differences at patient (breast, 638) or case (`prostate_ms`, 14) level, 95% cluster bootstrap, 5 seeds unless stated.

| Comparison | Task | Δ test DSC [95% CI] | Outcome | Source |
|---|---|---|---|---|
| **PRIMARY:** XFedGA vs FedGA (baseline) | breast | −0.0072 [−0.0109, −0.0035] | LOSE | [final tables](results/tables/final/comparisons.md) |
| **PRIMARY:** XFedGA vs FedGA (baseline) | prostate_ms | −0.0046 [−0.0082, −0.0015] | LOSE | [final tables](results/tables/final/comparisons.md) |
| Declared secondary: XFedGA+ vs FedGA | breast / prostate_ms | −0.0078 / +0.0061 | LOSE / TIE | [final tables](results/tables/final/comparisons.md) |
| g: XFedGA+ (DSC-only) vs FedGA, independent site RUNMC | prostate_ms_runmc | +0.0166 [+0.0084, +0.0254] | WIN | [g](results/site_g/g_runmc.md) |
| g2: XFedGA+ (DSC-only) vs FedGA, new seeds 6–10 | breast | +0.0121 [+0.0068, +0.0172] | WIN | [g2](results/replication/g2_breast.md) |
| Dose-response D1: slope XFedGA − FedGA (corrupted client) | breast, 3 seeds | +0.0197 [+0.0145, +0.0247] | supported | [dose-response](results/dose_response/dose_response.md) |
| Flower vs custom loop (XFedGA, XFedGA+) | breast, 3 seeds | 0 (bit-identical) | equivalent | [items b–f](results/post_items/post_items.md) |

More results (paper GA settings, H-ROB2, ablations, FedProx, bounds, sensitivity screens, explanation metrics) are in [results/post_items/post_items.md](results/post_items/post_items.md), [results/sensitivity/](results/sensitivity/) and [results/tables/final/comparisons.md](results/tables/final/comparisons.md). Figures: [results/figures/final/](results/figures/final/).

## Limitations
- **The pre-registered primary endpoint is LOSE on both tasks**, so XFedGA is not claimed to improve accuracy. XFedGA+ and its DSC-only control are post-v1 variants, declared as such. g2 reuses the same test patients (new seeds only).
- **`prostate_ms` has only 14 test cases** per held-out site; some secondary rows disagree between the bootstrap and the t-interval.
- **Explanation metrics depend on the definition:** EC (IoU at 0.5) and EC_mass can disagree in direction (`results/xai_ec/`).
- **Single 6 GB GPU:** batch size 8 instead of the paper's 128. Flower with 5 client processes runs out of memory, so the Flower demo and parity use 2 clients.
- **Declared deviations** (queue incidents, execution order, the superseded prostate setting) are recorded in `docs/protocol/plan.md`.
- **Results before 13 Oct 2026** are superseded: their checkpoints were selected on the test set (`results/legacy/README.md`, `docs/protocol/reaudit.md`).

## Tools
AI coding assistants were used to write code, run experiments and draft documentation, under the author's direction. The author made all research decisions and pre-registrations.
