# Data leakage in the original prostate setting, and its replacement

*Note for the thesis (methods / limitations). 18 Oct 2026. Every number comes from `results/audit/` and is produced by the scripts named below.*

## What was found
The original prostate federation used PROMISE12 (50 training cases, split into two clients) as client data and Medical Segmentation Decathlon Task05 (32 labelled cases) as the held-out server domain. That server data was split by case into fitness (8), validation (8) and test (16) sets.

**12 of the 50 PROMISE12 cases are the same MRI scans as Decathlon Task05 cases.**
- Pearson correlation of the image intensities is 1.000 at full resolution for every pair.
- The ground-truth masks differ slightly: Dice 0.89–1.00 for the labelled pairs. The two datasets were annotated separately.

8 of these 12 scans are in our server sets:

| Server set | Cases in set | Same scan as a client training case |
|---|---|---|
| fitness | 8 | 2 |
| validation | 8 | 2 |
| test | 16 | 4 |

The other 4 are in the Decathlon `imagesTs` folder, which we did not use. So the clients trained on 4 of the 16 test patients and 2 of the 8 validation patients. The "held-out" server domain was therefore not held out.

## How it was found
- **Trigger:** an inventory of a multi-site prostate collection (six sites), done before using it. Source: `analysis/multisite_inventory.py` → `reports/audit/multisite_inventory.md`.
  - That collection's RUNMC site matched Decathlon Task05 cases (30 of 30) and some PROMISE12 cases. This pointed to an overlap between PROMISE12 and Decathlon themselves.
- **Direct check:** `analysis/promise12_decathlon_overlap.py` → `reports/audit/promise12_decathlon_overlap.md`. Every PROMISE12 case was compared with every Decathlon case:
  1. an exact hash of the voxel array;
  2. a screen on volumes downsampled to 96 × 96 per slice, taking the best correlation over the 16 flips and transposes, for every pair with the same slice count;
  3. confirmation at full resolution, with the other volume resized to the PROMISE12 grid, plus the mask Dice.
- **Why it was missed earlier:** file names and formats differ between the datasets (`CaseNN.mhd` vs `prostate_NN.nii.gz`). Our earlier checks were split disjointness *within* each dataset, not image content *across* datasets.
- **Breast:** the same content-based audit was run on the breast datasets (`analysis/breast_duplicate_audit.py` → `reports/audit/breast_duplicates.md`). It found **no** duplicates between BUSI or BUS-UCLM (clients) and BUS-BRA (server), and none across BUS-BRA patients or splits. One limitation: copies cropped or padded differently are not detected by whole-image correlation.
  - **Limitation (kept as is):** there are duplicate images *inside* each breast client: BUSI has 26 duplicates and 70 near-duplicates, BUS-UCLM 8 and 9.
  - This is redundancy within one client's training data only; it never crosses into the server's fitness, validation or test data.
  - The images are not de-duplicated, because that would change the breast pipeline that the existing breast results were produced with.

## What is superseded
Every result of the PROMISE12 → Decathlon prostate setting is kept on disk and marked SUPERSEDED (`plan.md` §0.L). None is used for any claim. This includes:
- the prostate β screen;
- the prostate rows of the first headline comparison;
- the prostate explanation metrics, diagnostics and protocol-v2 screen.

The breast results are not affected.

## What replaces it
The main prostate experiment is now a **multi-site federation** (task `prostate_ms`): one client per site, with a different site held out as the server domain. It uses **no PROMISE12 or Decathlon data**.
- **Clients:** BIDMC, HK, I2CVB, RUNMC, UCL.
- **Server:** BMC, split by case into 8 fitness / 8 validation / 14 test cases in gland-size blocks.
- **Duplicate checks** (same method):
  - there are no duplicates between sites of the collection;
  - BMC and I2CVB do not occur in PROMISE12 or Decathlon;
  - BIDMC, HK and UCL are PROMISE12 scans, and RUNMC are Decathlon scans. That is harmless inside the self-contained experiment, but results on those sites are not independent of the superseded setting.
- **Pre-registration:** the design is pre-registered in `plan.md` before any run.
