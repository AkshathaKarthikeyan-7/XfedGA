# XFedGA — Session 1 audit (13 Oct 2026)

Scope: inventory, item-by-item verification of spec section 0, and extra discrepancies found along the way.
No existing file was modified. Evidence files produced this session:

| File | Produced by | Content |
|---|---|---|
| `results/audit/verify_checkpoints.json` | `analysis/verify_checkpoints.py` | re-evaluation of every root `*.pt` on the OOD split, matched to `round_history_*.csv` |
| `results/audit/existing_results.json` | `analysis/summarize_existing.py` | best-round / final / last-5 summaries of the existing CSVs |
| `results/timing/timing_summary.{csv,json}` | `analysis/summarize_timing.py` | per-run timing (from `results/timing/runs/*/timing_events.jsonl`, written by `analysis/timed_run.py`) |

Note: CLAUDE.md requires analysis scripts to live in `analysis/`. This session could only create files under `reports/audit/` and `results/timing/`, so the three scripts above sit there. Proposal: move them to `analysis/` in Session 2.

---

## 1. Inventory

### 1.1 Test suite, `fl_dist/`, pytest
| Check | Result |
|---|---|
| `tests/` in repo | **absent** |
| `fl_dist/` in repo | **absent** |
| `fl_dist`, `tests`, `conftest.py` anywhere on `D:\`, depth 6, excluding conda/anaconda envs | **none found** |
| pytest in `D:\conda_envs\fedga` | **not installed** (`importlib.util.find_spec('pytest') is None`); not installed this session |
| Other test-like files | `test_gradcam.py` (root) is a standalone CLI script, not a pytest test. pytest would still collect it by name, so exclude it or rename it once `tests/` exists |
| Note | A second env exists at `D:\anaconda3\envs\fedga`. CLAUDE.md names `D:\conda_envs\fedga`, and only that one was used |

### 1.2 Flower 1.7.0 readiness
- `from flwr.server import start_server`, `from flwr.client import start_numpy_client` and `from flwr.server.strategy import FedAvg` all import (flwr 1.7.0, grpcio 1.82.1).
- **Toy federation passed.** It ran from a scratch folder outside the repo: 1 server process plus 2 client processes, 1 round, a `torch.nn.Linear(4,1)` model, server on `127.0.0.1:18080`.
  - The server received both fit results with custom metrics `{'q_i': 0.1}` and `{'q_i': 0.2}`, which proves the fit-metrics dict can carry q_i.
  - Evaluate round: 2 results, 0 failures. Wall clock ≈ 17 s, mostly process start-up.
- **Port and firewall.** The port was free beforehand. No error appeared in any log. Binding to `127.0.0.1` normally does not trigger the Windows Defender firewall prompt, but this session cannot see GUI pop-ups, so **please confirm no prompt appeared on your screen at about 19:47**.
- **Warning.** `start_numpy_client()` is deprecated in 1.7.0; Flower suggests `start_client(client=NumPyClient().to_client())`. The spec mandates `start_numpy_client`. Both work in 1.7.0; the warning is cosmetic.

### 1.3 Nested `xfedga/` vs root
The nested folder is an **older (≈13 Aug), breast-only snapshot** of the root.

| Path | Status | Difference |
|---|---|---|
| README.md, requirements.txt, configs/breast_clients.yaml, configs/environment.md, data/breast_dataset.py, debug_busbra.py, diagnose_class_balance.py, experiments/train_baseline_fedavg.py, fedga/{fedavg,ga_optimizer,losses}.py, inspect_model.py, models/residual_unet.py, test_gradcam.py | identical | – |
| experiments/train_fedga.py | differs | nested `compute_client_saliency_score(..., max_samples=20)`; root uses `max_samples=40` (line 144). Otherwise identical |
| data/server_split.py | differs | nested lacks `get_prostate_server_split` |
| data/prostate_dataset.py | differs | older version: plain min-max normalisation with no 1st/99th-percentile clipping, and longer docstrings |
| round_history_base_fedga.csv, round_history_xfedga_fedga.csv | differ | older breast runs: DSC column only, no EC/PG |
| round_history_fedavg_baseline.csv, global_model_fedavg_baseline{,_BEST}.pt | identical | – |
| global_model_{base,xfedga}_fedga{,_BEST}.pt | differ | older breast runs (13 Aug) |
| missing in nested | – | `xai/` (so nested `train_fedga.py` cannot run standalone), `scripts/`, `train_fedga_prostate.py`, prostate config, analysis scripts |
| `xfedga/raw data/BUS-BRA/BUSBRA/` | nested only | partial copy: 816 images, `bus_data.csv`, `LICENSE.txt`; no `Masks/` |

**Unique content worth keeping:** only the historical 13-Aug breast results: two CSVs and four checkpoints, from the q_i `max_samples=20` era. They are a record of what the Review-1 deck was based on. Proposal: move them to `results/legacy_ood_selected/2026-08-13/` with a README, then delete the folder after your approval. No unique code exists there.

### 1.4 Datasets on disk
| Dataset | Path | Role | On disk | Used by the code |
|---|---|---|---|---|
| BUSI | `raw data/BUSI` | client 1 (breast) | 780 images (437 benign, 210 malignant, 133 normal); 18 extra `_mask_1/_mask_2` lesion masks | 647 non-empty → **518 train** (80%); the 20% client val is discarded |
| BUS-UCLM | `raw data/BUS-UCLM/.../BUS-UCLM/{images,masks}` | client 2 (breast) | 683 images, 38 patient prefixes (`INFO.csv`: 419 normal, 174 benign, 90 malignant), all 856×606 | **174 non-empty → 140 train**; see D-1 |
| BUS-BRA | `raw data/BUS-BRA/BUSBRA/{Images,Masks}` | server (breast) | **1,875 images from 1,064 patients** (`bus_data.csv` column `Case`); 4 devices (GE Logiq 7: 902, GE Logiq 5: 809, Toshiba Aplio 300: 139, U-Systems: 25); 1,268 benign, 607 malignant; native widths 240–579 px | 938 fitness / 937 OOD, **image-level split**; see D-2 |
| PROMISE12 | `raw data/PROMISE12` | clients 1+2 (prostate) | **50 labelled cases** (Case00–49, `.mhd/.raw` + `_segmentation`) | 25/25 random case split → 379 / 399 non-empty slices |
| Decathlon Task05 | `raw data/Task05_Prostate/Task05_Prostate` | server (prostate) | **32 labelled cases** (`imagesTr`/`labelsTr`, 4-D T2+ADC, labels PZ/TZ) + 16 unlabelled `imagesTs`; 10 macOS `._*` junk files (filtered by the indexer); `dataset.json` licence CC-BY-SA 4.0 | 16 fitness cases (242 slices) / 16 OOD cases (233 slices) |

### 1.5 Panel decks
- `D:\Amrita\Project\Literature review\Panel review -1\XFedGA_Panelreview_1.pptx` (20 slides) and `XFedGA_Panel_Review_1.docx`
- `D:\Amrita\Project\Literature review\Panel review -2\XFedGA_Panel_Review_2.pptx` (19 slides)

---

## 2. Section 0, item by item

Legend: **CONFIRMED** (file:line) · **DIFFERENT** (what the code actually does) · **NOT FOUND**.

### 2.1 Files and architecture
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Listed files exist | CONFIRMED | All present. Extra items not in the list: nested `xfedga/`, empty `notebooks/`, `gradcam_figures*/`, root PNGs, `README.md`, `requirements.txt`, 10 root `*.pt`. Git history: `c44be6b initial snapshot + spec`, `7fcdfae spec: fl_dist and tests not yet built` |
| 2 | Single-process, single-GPU, sequential client loop | CONFIRMED | `experiments/train_fedga.py:237-247`, `experiments/train_fedga_prostate.py:107-117` |
| 3 | MONAI provides the model and HausdorffDistanceMetric | CONFIRMED | `models/residual_unet.py:13-20` (UNet, 1,625,161 params, InstanceNorm2d, no running stats); `fedga/ga_optimizer.py:18` |
| 4 | `fl_dist/` does not exist | CONFIRMED | §1.1 |
| 5 | No `tests/`, no test suite | CONFIRMED | §1.1 |
| 6 | "fl_dist, 35 tests, ablations A–E, FedProx, server fine-tune already built" | NOT FOUND (the claim is false) | grep for `fedprox`, `proximal`, `fine.?tune`, `finetune`: 0 hits. Only modes `base` and `xfedga` exist (`train_fedga.py:181`) |
| 7 | Nested `xfedga/` is an older breast-only duplicate | CONFIRMED, with differences | §1.3 |

### 2.2 Tasks and data
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 8 | Breast clients are BUS-UCLM and BUSI; server is BUS-BRA | CONFIRMED | `configs/breast_clients.yaml` (clients BUSI, BUS-UCLM; server BUS-BRA). The README still lists BUS-BRA as a client and I2CVB as planned (stale) |
| 9 | PROMISE12 split randomly 50/50 by case into 2 clients | CONFIRMED | `data/prostate_dataset.py:141-148`; 25/25 cases |
| 10 | Prostate server data is Decathlon Task05 | CONFIRMED | `configs/prostate_clients.yaml`, `data/server_split.py:49-61` |
| 11 | Decathlon has only 32 labelled cases | CONFIRMED | 32 `imagesTr` / 32 `labelsTr` (non-`._`) |

### 2.3 Defaults
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 12 | 15 rounds, 10 local epochs, Adam lr 1e-3, batch 8, population 6, generations 5, pc 0.5, pm 0.1, sigma0 0.01 | CONFIRMED for both FedGA scripts | `train_fedga.py:182-190`, `train_fedga_prostate.py:52-60`. Additional details: Adam betas (0.9, 0.99) (`train_fedga.py:49`); a fresh optimizer every round. All values are CLI defaults; none come from YAML |
| 13 | (FedAvg baseline) | DIFFERENT | `train_baseline_fedavg.py:68` defaults to **20** rounds; the CSV has 15 rows, so it was run with `--rounds 15`. The script is breast-only and duplicates `local_update`/`evaluate_dsc` |
| 14 | tau | NOT IN CONFIG | `tau=10.0` is hard-coded (`fedga/ga_optimizer.py:89,95`); not exposed in the CLI |

### 2.4 Aggregation
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 15 | alpha_i ∝ (n_i/N)·max(q_i, 1e-3)^beta, normalized | CONFIRMED | `fedga/fedavg.py:25-31` |
| 16 | beta defaults to 0.5 | CONFIRMED | `fedga/fedavg.py:13` |
| 17 | Rationale "q_i from about 20 images, swings 0.04–0.20" | DIFFERENT | The rationale text is in the docstring (`fedga/fedavg.py:19-23`), but the root code uses **40** samples (`train_fedga.py:144`); 20 was the nested/older value. The logged prostate q_i range is **0.19–0.73**, with a mean round-to-round change of ≈0.09 (`existing_results.json`). Breast q_i was never logged. No beta=1.0 run log exists in the repo, so the "beta=1.0 unstable" claim has no evidence on disk |
| 18 | Prostate script has `--beta` | CONFIRMED | `train_fedga_prostate.py:61`, used at `:120` |
| 19 | Breast script has no `--beta` | CONFIRMED | `train_fedga.py:178-191`; the call at `:250` uses the default 0.5 |
| 20 | No post-GA blend | CONFIRMED | `train_fedga.py:260`, `train_fedga_prostate.py:130`: `global_model.load_state_dict(best_candidate_state)` |
| 21 | (beta used for the existing prostate XFedGA run) | UNVERIFIABLE | The Review-2 deck (slide 6) says "β = 0.5 breast, **0.2 prostate**". The script default is 0.5, and beta is not logged in the CSV or the checkpoint name |

### 2.5 q_i
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 22 | q_i = `compute_client_saliency_score`, client-side, on the client's own training loader | CONFIRMED | `train_fedga.py:243-245`; computed on the client's local model after training |
| 23 | max 20 samples | DIFFERENT | **40** (`train_fedga.py:144`); 20 only in nested `xfedga/` |
| 24 | uses `explanation_consistency(cam, mask, threshold=0.5)` | CONFIRMED | `train_fedga.py:158` |
| 25 | fallback 0.5 | DIFFERENT (partial) | 0.5 is returned only when **no** sample was processed (`:164`). A Grad-CAM exception on an image appends **0.0** (`:160`), which pulls q_i down rather than falling back |
| 26 | Loaders are shuffled, so q_i uses different images each round | CONFIRMED | `shuffle=True` (`train_fedga.py:209`, `train_fedga_prostate.py:79-80`); the first 40 images of a fresh shuffle are used |

**Exact definition, as computed:**
q_i = (1/K)·Σ_{k=1..K} EC(CAM_k, M_k), with K = min(40, |D_i|). The images are the first K of a freshly shuffled pass over client i's training set.
- CAM_k is the Grad-CAM (pytorch-grad-cam 1.5.0) of the class-1 logit summed over **all** pixels (`SemanticSegmentationTarget(1, ones)`, `xai/explainability.py:42`).
- It is taken at decoder layer `model.model[1].submodule[2][1].conv.unit0.conv` (16 channels; `train_fedga.py:44`), upsampled and min-max normalised per image by the library.

### 2.6 GA
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 27 | Population = one unchanged w_agg + (P−1) Gaussian copies at sigma0 | CONFIRMED | `fedga/ga_optimizer.py:29-33` |
| 28 | Initial population is never annealed | CONFIRMED | `run_ga` passes `sigma0` (`:96`) |
| 29 | Rank, keep top half, uniform crossover (pc), sparse Gaussian mutation (pm) | CONFIRMED | `:98-117`. Details: parents are drawn uniformly **with replacement** from the top half (`:109`, so p1 = p2 is possible). With probability 1−pc the child is a copy of p1. Crossover is a per-element 50/50 mask (`:40`). Every offspring is mutated: per-element Bernoulli(pm) mask × N(0, sigma²) (`:45-52`). This covers all float tensors, including PReLU weights; there are no BN buffers |
| 30 | sigma = sigma0·exp(−g/10) in xfedga mode; fixed in base | CONFIRMED | `:89-90, :99`. g = 0…4, so the last offspring use sigma0·e^−0.4 = 0.0067 |
| 31 | About 36 fitness evaluations per round | CONFIRMED (measured: 36) | P·G + P = 6·5 + 6. The top half is **re-evaluated** every generation with no cache, so only P + G·P/2 = 21 candidates are distinct; **15 of 36 evaluations (42%) are redundant** |
| 32 | (Elitism guarantee) | CONFIRMED by reading; untested | The best-ranked individual always survives (`:105`), and the final argmax (`:119-121`) runs over a population containing it. So fitness(winner) ≥ fitness(w_agg) holds **if evaluation is deterministic**. No unit test exists |

### 2.7 Fitness
| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 33 | F = 0.5·DSC − 0.3·HD95_norm + 0.2·EC | CONFIRMED | `fedga/ga_optimizer.py:83-86`; the caller passes no weights (`train_fedga.py:141`) |
| 34 | Weights hard-coded, not in config, no tuning evidence | CONFIRMED | – |
| 35 | HD95 normalized by 384, clipped to [0,1] | CONFIRMED | `:62, :80`. If both masks are empty → 0; if exactly one is empty → **1.0** (`:71-74`) |
| 36 | (Real image size vs 384) | DIFFERENT in meaning | All inputs are resized to **384×384** (`data/breast_dataset.py:17,149`, `data/prostate_dataset.py:63,155`, `image_size: [384,384]` in both YAMLs). 384 is the image **side**, not the diagonal (543 px), although the docstring says "diagonal-ish". Units are pixels of the resized grid, not mm. Aspect ratio is not preserved (BUS-UCLM 856×606, BUS-BRA 240–579 px wide). For prostate, the 0.5 mm resampling (`prostate_dataset.py:26`) is undone by the fixed 384 resize, so the physical pixel size varies by case |
| 37 | XFedGA scores a 40-image subsample with Grad-CAM; base scores the whole fitness set | CONFIRMED | `train_fedga.py:105, 172-174`. The 40 are the **first 40 of the unshuffled fitness loader**, the same for all candidates and rounds. For prostate that is the first ~3 of 16 fitness cases. Base mode scores all 938 breast images / 242 prostate slices |
| 38 | Base mode = DSC-only fitness, fixed sigma, plain FedAvg | CONFIRMED | `train_fedga.py:174`, `ga_optimizer.py:99`, `train_fedga.py:252` |

### 2.8 `explanation_consistency`
**CONFIRMED: it is IoU.** `xai/explainability.py:17-27`:

EC(S, M) = |{p : S(p) ≥ 0.5} ∩ {p : M(p) = 1}| / |{p : S(p) ≥ 0.5} ∪ {p : M(p) = 1}|, and 0 if the union is empty.

- S is the per-image min-max-normalised Grad-CAM, so the threshold is relative to that image's own peak.
- Exact name to use: **"Grad-CAM–mask IoU at threshold 0.5"**. It measures plausibility (overlap with the annotation), not faithfulness.

### 2.9 Gradient-aware perturbation
**NOT FOUND.**
- No code uses gradients to shape a perturbation. grep for `grad-aware`, `gradient-guided`, `.grad`, `autograd`, `perturb` finds nothing outside Grad-CAM.
- Mutation is `torch.randn_like` under a Bernoulli mask (`fedga/ga_optimizer.py:45-52`). Gradients are used only inside Grad-CAM, to score explanations.
- The phrase does not appear in the Review-1 or Review-2 decks either; the only "gradient" hit describes FedATA.
- Per the spec, it must not appear in the novelty statement unless it is implemented and ablated (Tier 2).

### 2.10 Existing results (all superseded)
**Selection rule of every `*_BEST.pt`: CONFIRMED to be argmax of OOD DSC over rounds.** Code: `train_fedga.py:267`, `train_fedga_prostate.py:137`, `train_baseline_fedavg.py:127`. Re-evaluating each checkpoint reproduces its CSV row exactly (|Δ| ≤ 3e-8, `verify_checkpoints.json`):

| Task | Method | BEST = round | OOD DSC at BEST | Final round (15) DSC | Last-5 mean ± std |
|---|---|---|---|---|---|
| Breast | FedAvg | 1 | 0.6317 | 0.5355 | 0.5348 ± 0.0122 |
| Breast | Base FedGA | 2 | 0.6462 | 0.5752 | 0.5648 ± 0.0107 |
| Breast | XFedGA | 2 | 0.6503 | 0.5452 | 0.5380 ± 0.0046 |
| Prostate | Base FedGA | 8 | 0.5439 | 0.4254 | 0.4465 ± 0.0219 |
| Prostate | XFedGA | 2 | 0.4960 | 0.3632 | 0.3696 ± 0.0223 |

Source: `existing_results.json`, built from the root `round_history_*.csv`. These are image/slice-level means on the OOD half, from single unseeded runs.

- **Every method's OOD DSC peaks in rounds 1–2 and then declines.** Best-round selection on OOD data therefore turns a downward trend into a headline number.
- **On the last-5-round mean, XFedGA is below base FedGA on both tasks.** These are single unseeded runs, so this is not evidence either way. But the earlier "no accuracy cost" reading depends on OOD-selected checkpoints.
- **Prostate numbers in spec §0.**
  - DSC 0.544 vs 0.496: CONFIRMED from the CSVs and checkpoints.
  - 233 slices / 16 cases: CONFIRMED.
  - HD95_norm 0.243 vs 0.215, EC 0.120 vs 0.197, PG 60.9% vs 57.5% (p = 0.34): NOT FOUND in any repo file. They were probably console output of `compare_significance_prostate.py`, which saves nothing.
- **Breast numbers.** Found only in the decks: Review-2 slide 10 and Review-1 slide 14.
  - n = 937; Base FedGA vs XFedGA: DSC 0.646 vs 0.650 (p = 0.30), HD95_norm 0.280 vs 0.263 (p < 1e-4), EC 0.145 vs 0.153 (p = 0.0066), PG 57.6% vs 71.7% (p < 1e-4).
  - Only the DSC values trace to a repo file (CSVs above). They came from the OOD-selected BEST checkpoints with per-image tests on 937 correlated images (see D-2).

### 2.11 Known weaknesses 1–10
| # | Weakness | Verdict |
|---|---|---|
| 1 | Best round selected on the OOD test set | CONFIRMED for **prostate and breast, and for the FedAvg baseline** (§2.10) |
| 2 | Annealing barely decays (0.01 → 0.0067) | CONFIRMED (§2.6, #30) |
| 3 | EC is optimised and reported (circular) | CONFIRMED: fitness `train_fedga.py:129`; reported `:93` |
| 4 | Prostate partition near-IID | CONFIRMED as random; but PROMISE12 has strong acquisition clusters (§3, D-6) |
| 5 | Base vs XFedGA differ in several things at once | CONFIRMED: aggregation, fitness, annealing **and** fitness-subsample size (4 factors) |
| 6 | Full fitness set vs 40 images | CONFIRMED (#37) |
| 7 | Shuffled client loaders → q_i on different images each round | CONFIRMED (#26) |
| 8 | Hand-set fitness weights | CONFIRMED (#34) |
| 9 | Decathlon has 32 labelled cases | CONFIRMED |
| 10 | Metric name | CONFIRMED to be IoU (§2.8) |

### 2.12 Methods the existing scripts can run
| Method | Runnable now? | How |
|---|---|---|
| FedAvg (breast) | yes | `experiments/train_baseline_fedavg.py --rounds 15` |
| FedAvg (prostate) | yes, via a CLI trick (not run) | `train_fedga_prostate.py --mode base --population 1 --generations 0`: `run_ga` returns w_agg after 1 wasted DSC evaluation (`ga_optimizer.py:29-33, 97, 119-121`) |
| A: base FedGA | yes | `--mode base` |
| B: weighted aggregation only, no GA | yes, via the same trick (not run) | `--mode xfedga --population 1 --generations 0`. Breast beta is fixed at 0.5 |
| C: FedAvg + GA(multi-objective) | **no** | needs separate aggregation / fitness / annealing switches |
| D: FedAvg + GA(DSC) + annealing | **no** | same |
| E: full XFedGA | yes | `--mode xfedga` |
| F: XFedGA with random q_i | **no** | needs a q_i-source switch |
| FedProx | **no** | NOT FOUND |
| Server fine-tune | **no** | NOT FOUND |
| Centralized / local-only | **no** | NOT FOUND |
| Random-search control | **no** | NOT FOUND |
| NSD | **no** | NOT FOUND |
| Insertion/deletion AUC | code exists, **never called** | `xai/explainability.py:81` |

A then differs from E in four factors: aggregation, fitness, annealing and subsample. A clean ablation needs independent switches: `aggregation ∈ {fedavg, saliency}`, `fitness ∈ {dsc, multi}`, `anneal ∈ {off, on}`, `q_source ∈ {gradcam, random, constant}`, `search ∈ {ga, random, none}`, plus one fixed fitness subsample shared by all.

---

## 3. PROMISE12 site/vendor metadata
**None on disk.**
- The folder holds only `CaseNN.{mhd,raw}` and `CaseNN_segmentation.{mhd,raw}` (0 other files).
- All 100 `.mhd` headers carry the same 13 keys: ObjectType, NDims, BinaryData, ByteOrderMSB, CompressedData, TransformMatrix, Offset, CenterOfRotation, AnatomicalOrientation, ElementSpacing, DimSize, ElementType, ElementDataFile. There is no manufacturer, site or field-strength tag.

The headers do show **distinct acquisition profiles**, in contiguous blocks of case IDs (in-plane spacing / matrix):

| Profile | Cases | n |
|---|---|---|
| fine: 0.27–0.35 mm, 512² | 00–12, 33, 36 | 15 |
| mid: 0.39 mm, 512² | 26–32 | 7 |
| coarse: 0.47–0.75 mm, 256²–384² | 13–25, 34, 35, 37–49 | 28 |

Whether these blocks correspond to acquisition centres is **an inference, not documented on disk**. They are a defensible basis for the heterogeneous ("acquisition-profile skew") partition in spec §2, provided they are described as acquisition-profile groups, not sites.

---

## 4. Additional discrepancies (not in spec §0)

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| **D-1** | **high** | **All 90 malignant BUS-UCLM lesions are silently dropped.** BUS-UCLM masks are colour-coded: benign = pure green (BGR 0,255,0), malignant = pure red (BGR 0,0,255). The loader reads masks as grayscale and thresholds at >127. Green → 150 (kept); red → 76 (**empty**), so `filter_nonempty_masks` then removes those images. Client 2 trains on **benign lesions only** (174 = exactly the benign count) | `data/breast_dataset.py:19,28`; per-label check over all 683 masks |
| **D-2** | **high** | **The BUS-BRA fitness/OOD split is per image, not per patient.** 1,875 images come from 1,064 patients; **382 patients appear in both halves**, which accounts for 382 of the 937 OOD images (41%). This violates spec §2. The per-image tests on "n = 937" treat correlated images as independent | `data/server_split.py:21-31`; `bus_data.csv` column `Case` |
| D-3 | medium | No seeding anywhere: no `manual_seed` / `np.random.seed`. The GA (numpy + torch RNG), client shuffles and initial weights are unseeded. Only the data splits use seed 42 | grep |
| D-4 | medium | The per-round OOD explainability log uses only the **first 30** OOD samples (prostate: ≈2 cases) | `train_fedga.py:73` |
| D-5 | medium | The XFedGA fitness subsample is the first 40 fitness images. For prostate that is ≈3 of 16 cases, not a stratified sample | `train_fedga.py:105-136` |
| D-6 | medium | PROMISE12 has 3 acquisition-profile blocks (§3); the random split ignores them | – |
| D-7 | low | BUSI multi-lesion images: the extra `_mask_1`/`_mask_2` files (18) are ignored, so ground truth is incomplete for those images | `data/breast_dataset.py:50-52` |
| D-8 | low | The breast clients' 20% validation split is computed and discarded | `train_fedga.py:207` |
| D-9 | low | 42% of GA evaluations are redundant (elites re-scored). Caching them is a free ≈40% cut in GA time, **if** evaluation is deterministic | §2.6 #31 |
| D-10 | low | Docs are stale: README (BUS-BRA as client, I2CVB, Attention U-Net for prostate), `configs/environment.md` (cu121, ≥8 GB VRAM). `requirements.txt` is unpinned and misses flwr, SimpleITK, nibabel, scipy | – |
| D-11 | low | Statistics scripts use per-image Shapiro-gated t-test/Wilcoxon and save nothing to disk | `compare_significance*.py` |
| D-12 | info | Data loading is single-threaded (`num_workers=0`): the GPU sits at 26–46% while total CPU is ~75%, so the loop is CPU-bound (nvidia-smi sample during the prostate base run) | §5 of `plan.md` |
