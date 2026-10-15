# Session 1 status: COMPLETE, awaiting approval (15 Oct 2026)

All Session-1 items are done: `plan.md` is finished and the end-of-session report is `session1_report.md`. The budget is confirmed as 4 h per weekday and 10 per weekend day = 104 GPU-h (23 Oct – 8 Nov). The spec's "pull (7)" means item (8), the explanation metrics, which are pulled into Tier 1. No code changes until you approve `plan.md` and Q1–Q10.

---
(Previous pause log below, kept for the record.)

# Session 1 status: paused 14 Oct 2026 (~20:10 IST, battery)

No python process running; GPU has no compute apps (checked). No existing repo file modified; all new files are under `reports/audit/` and `results/timing/`. Spec read at commit `766c8cc` (`fl_dist` now in Tier 1).

## Done (all measurements complete)
| Item | Output |
|---|---|
| Inventory, section-0 verification, discrepancies D-1…D-12 | `audit.md` |
| BEST checkpoints are OOD-selected (exact re-evaluation) | `verify_checkpoints.json` |
| Existing results summary | `existing_results.json` |
| Paper comparison (pypdf, page refs; naming "FedGA (baseline)"; setup table; p.10 note) | `fedga_paper_comparison.md` |
| Timing, prostate FedGA: 15/15 rounds, **measured**, 131 s/round, 0.55 h | `results/timing/runs/full_prostate_base` |
| Timing, prostate XFedGA: 8/15 rounds (stopped by you on 13 Oct), 210 s/round, **15-round total extrapolated** at 0.88 h. Round 7 inflated by parallel CPU work | `runs/full_prostate_xfedga` |
| Timing, breast XFedGA: 15/15 rounds, **measured**, 223 s/round, 0.94 h (the round-3 projection was < 1 h, so it ran in full) | `runs/full_breast_xfedga` |
| Timing, breast FedGA (old protocol: full 938-image fitness set): 3 rounds, 394 s/round, **15-round total extrapolated** at 1.66 h | `runs/full_breast_base` |
| GPU-sharing test: 2 prostate FedGA runs in parallel, 2 rounds. 248 s/round each vs 118 s single (rounds 1–2), so **throughput factor 0.95: no gain**. Local training slowed ~3×, consistent with a CPU-bound loop; 712 MiB reserved per process | `runs/share2_prostate_base_p{1,2}` |
| Batch-size memory check (1 forward/backward): batch ≤ 64 fits the 6 GB of dedicated VRAM (64 → 4.8 GB reserved); 96 → 7.2 GB and **128 → 9.5 GB exceed it** (no OOM only because Windows spills into shared memory) | `results/timing/batch_memcheck.json` |
| Per-case variance (old fitness half only): prostate per-case SD of paired ΔDSC 0.070 → 95% CI ±0.038 with 16 test cases; breast per-patient SD 0.137 → ±0.011 with 600 patients | `case_level_variance.json` |
| Cost model run on final timings | `analysis/plan_costs.py` → `plan_costs.{json,md}` |
| `plan.md`, partly written: §1 discrepancies, §2 splits (Decathlon 8/8/16), §6 build effort, §7 `fl_dist/` design, §9 hypotheses | `plan.md` |
| Timing summary regenerated | `results/timing/timing_summary.{csv,json}` |

Key cost numbers (from `plan_costs.json`):
- Tier 1 = 79.7 GPU-h (required 50.5 + cuttable 29.2); 103.6 h with a ×1.3 contingency.
- Tier 2 = 82.5 GPU-h; optional paper-GA-settings run = 4.9 h.
- β ablation: full grid 26.7 h; reduced grid 16.0 h; screening design 15.5 h.
- Budget 15 Oct–8 Nov, after a build gap until 22 Oct: 184 h at 8/16 h per day, 104 h at 4/10, 72 h with weekends only (12 h).

## Pending
1. Fill the `plan.md` placeholders `<!-- TIMINGS -->`, `<!-- COSTS -->`, `<!-- BUDGET -->`, `<!-- CUTS -->` and `<!-- DECISIONS -->` from `timing_summary.json` and `plan_costs.{json,md}` (no new runs needed).
2. Fix the "–" encoding in `plan_costs.md`: the script now writes UTF-8; just rerun it.
3. End-of-session report (CLAUDE.md format).
4. Then stop for your approval. **No code changes** until then.

## Resume commands (PowerShell, from `D:\new repo\xfedga`)
```powershell
$py = "D:\conda_envs\fedga\python.exe"
& $py results/timing/summarize_timing.py   # regenerate timing summary (no GPU)
& $py analysis/plan_costs.py          # regenerate cost tables (no GPU)
```
Then ask Claude Code to fill the `plan.md` placeholders and write the end-of-session report. No further GPU runs are needed for Session 1.

## Still open for you
- Compute budget: which scenario is realistic (8/16, 4/10 or weekends-only)? Yesterday you could not leave the laptop on overnight.
- Spec numbering: "If the timings allow, pull (7) into Tier 1. Without independent explanation metrics…". Explanation metrics are now (8). I assume (8) is meant.
