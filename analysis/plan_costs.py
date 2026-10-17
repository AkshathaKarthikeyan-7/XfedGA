"""
Cost model for the Session-1 plan. Reads measured timings from
results/timing/timing_summary.json (+ GPU-sharing runs) and writes
results/audit/plan_costs.json and results/audit/plan_costs.md.

Every per-run cost is built from MEASURED per-round components:
  L   local training, both clients (s/round)
  Q   q_i computation, both clients (s/round)
  e_m one multi-objective fitness evaluation on the 40-image subsample (s)
  d   DSC-only evaluation cost per server image (s/image) = base s_per_eval / n_old_fitness_images
  V   per-round server evaluation (old OOD DSC + 30-image Grad-CAM logging) (s)
  S   setup (data indexing) (s)
combined with ASSUMPTIONS that are listed in ASSUMPTIONS below and in the output.

Usage (from repo root):
    python analysis/plan_costs.py
"""
import datetime as dt
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, "results", "audit")  # output folder
TIMING = os.path.join(REPO, "results", "timing", "timing_summary.json")

ROUNDS = 15
EVALS = 36                 # P*G + P with P=6, G=5 (current code, no elite cache)
EVALS_CACHED = 21          # P + G*P/2 if elite fitness is cached (audit D-9)
N_FIT_SUB = 40             # fixed fitness subsample, shared by all GA methods (spec section 2)
OLD_FIT_IMAGES = {"prostate": 242, "breast": 938}       # old fitness halves (smoke/full-run logs)
NEW_FIT_IMAGES = {"prostate": 121, "breast": 282}       # proposed fitness sets (plan.md section 2)
TEST_EVALS_PER_RUN = 1     # test set evaluated once per run (cost ~ one V)
CONTINGENCY = 1.3          # packing losses (4-h weekday slots), reruns, smoke runs
BUDGET = (4, 10)           # GPU-h per weekday / weekend day, confirmed 15 Oct 2026
SLOWDOWNS = [1.0, 1.25, 1.5]   # real runs slower than estimated, on top of contingency
# Tier 1 fill order: items are kept from the top; cut from the bottom if the budget runs out.
FILL_ORDER = ["T1-flsmoke", "T1-beta", "T1-core-fedavg", "T1-core-A", "T1-core-E", "T1-parity",
              "T1-abl-B", "T1-abl-F", "T2-expl", "T1-core-fedprox", "T1-abl-RS", "T1-abl-C",
              "T1-fullfit", "T1-abl-D", "T1-bounds"]

ASSUMPTIONS = [
    "Per-round validation cost = measured per-round OOD-eval cost V (val sets are <= old OOD half): conservative.",
    "FedProx local cost = FedAvg local cost (proximal term overhead unmeasured; expected small).",
    "Centralized and local-only bounds cost = FedAvg run (same total local epochs over the same data); local-only evaluates 2 models -> 2V.",
    "FedGA (baseline) in the new protocol scores the same fixed 40-image fitness subsample as XFedGA: 36 x 40 x d per round.",
    "Random-search control matches XFedGA/C: 36 multi-objective evaluations per round, no q_i.",
    "fl_dist run = 1.15 x custom-loop run (process start-up + gRPC serialisation of 6.5 MB per client per direction; unmeasured).",
    "Explanation-metric evaluation per test image = 12 Grad-CAM-equivalents + 42 forward passes (2 CAM methods, deletion+insertion 21 steps each, 10-layer randomisation check).",
    "GPU-sharing factor applies to runs launched in parallel on one GPU; measured on prostate FedGA only and assumed to hold for other run types.",
]


def load_timing():
    recs = {r["run"]: r for r in json.load(open(TIMING))}
    comp = {}
    for task in ["prostate", "breast"]:
        b, x = recs.get(f"full_{task}_base"), recs.get(f"full_{task}_xfedga")
        if not (b and x):
            comp[task] = None
            continue
        comp[task] = {
            "L": (b["mean_local_train_s"] + x["mean_local_train_s"]) / 2,
            "Q": x["mean_qi_s"],
            "e_m": x["s_per_fitness_eval"],
            "d": b["s_per_fitness_eval"] / OLD_FIT_IMAGES[task],
            "V": (b["mean_ood_eval_s"] + x["mean_ood_eval_s"]) / 2,
            "S": max(b["setup_s"], x["setup_s"]),
            "measured_round_base_s": b["mean_wall_s"], "measured_round_xfedga_s": x["mean_wall_s"],
            "base_rounds_measured": b["rounds_completed"], "xfedga_rounds_measured": x["rounds_completed"],
            "base_extrapolated": b["extrapolated"], "xfedga_extrapolated": x["extrapolated"],
            "base_status": b["status"], "xfedga_status": x["status"],
        }
    share = [r for k, r in recs.items() if k.startswith("share2_prostate_base")]
    sharing = None
    if len(share) == 2 and recs.get("full_prostate_base"):
        # reference = rounds 1-2 of the single prostate FedGA run (same rounds as the parallel runs)
        ev = os.path.join(REPO, "results", "timing", "runs", "full_prostate_base", "timing_events.jsonl")
        r12 = [e["wall_s"] for e in map(json.loads, open(ev)) if e["event"] == "round" and e["round"] <= 2]
        single = sum(r12) / len(r12)
        par = sum(r["mean_wall_s"] for r in share) / 2
        sharing = {"single_round_s_rounds1_2": single, "parallel_round_s_each": par,
                   "throughput_factor": 2 * single / par,
                   "peak_reserved_MiB_each": [r["peak_reserved_MiB"] for r in share]}
    return comp, sharing


def round_cost(c, method, task):
    L, Q, e, d, V = c["L"], c["Q"], c["e_m"], c["d"], c["V"]
    g_sub = EVALS * e
    a_sub = EVALS * N_FIT_SUB * d
    table = {
        "fedavg": L + V, "fedprox": L + V, "centralized": L + V, "local_only": L + 2 * V,
        "A_fedga": L + a_sub + V, "B_weighted_only": L + Q + V, "C_fedavg_ga_multi": L + g_sub + V,
        "D_fedga_anneal": L + a_sub + V, "E_xfedga": L + Q + g_sub + V, "F_random_qi": L + g_sub + V,
        "random_search": L + g_sub + V,
        "A_full_fitness_set": L + EVALS * NEW_FIT_IMAGES[task] * d + V,
        "E_full_fitness_set": L + Q + EVALS * e * NEW_FIT_IMAGES[task] / N_FIT_SUB + V,
        "fedga_paper_ga_settings": L + 210 * NEW_FIT_IMAGES[task] * d + V,  # P=10,G=20 -> 210 evals
    }
    return table[method]


def run_hours(c, method, task, rounds=ROUNDS, factor=1.0):
    return factor * (c["S"] + rounds * round_cost(c, method, task) + TEST_EVALS_PER_RUN * c["V"]) / 3600


def expl_eval_hours(c, n_images):
    g = c["e_m"] / N_FIT_SUB
    return n_images * (12 * g + 42 * c["d"]) / 3600


# (id, tier, priority, label, task(s), method, seeds, rounds, factor)
EXPERIMENTS = [
    ("T1-beta", 1, "1b", "beta ablation, beta in {0,0.2,0.25,0.5,1.0}", "both", "E_xfedga", 5 * 3, ROUNDS, 1.0),
    ("T1-flsmoke", 1, "2", "fl_dist 1-round smoke run + message audit", "both", "E_xfedga", 1, 1, 1.15),
    ("T1-core-fedavg", 1, "3", "FedAvg", "both", "fedavg", 5, ROUNDS, 1.0),
    ("T1-core-A", 1, "3", "FedGA (baseline)", "both", "A_fedga", 5, ROUNDS, 1.0),
    ("T1-core-E", 1, "3", "XFedGA at selected beta (3 seeds reused from beta ablation)", "both", "E_xfedga", 2, ROUNDS, 1.0),
    ("T1-core-fedprox", 1, "3*", "FedProx (spec section 5 baseline; not named in tier list)", "both", "fedprox", 5, ROUNDS, 1.0),
    ("T1-parity", 1, "4", "fl_dist XFedGA, breast (custom seeds reused)", "breast", "E_xfedga", 3, ROUNDS, 1.15),
    ("T1-abl-B", 1, "5", "B weighted aggregation only", "both", "B_weighted_only", 3, ROUNDS, 1.0),
    ("T1-abl-F", 1, "5", "F XFedGA with random q_i", "both", "F_random_qi", 3, ROUNDS, 1.0),
    ("T1-abl-C", 1, "5c", "C FedAvg + GA(multi-objective)", "both", "C_fedavg_ga_multi", 3, ROUNDS, 1.0),
    ("T1-abl-D", 1, "5c", "D FedGA + annealing", "both", "D_fedga_anneal", 3, ROUNDS, 1.0),
    ("T1-abl-RS", 1, "5c", "random-search control (matched evaluations)", "both", "random_search", 3, ROUNDS, 1.0),
    ("T1-fullfit", 1, "5c", "full-fitness-set check (A and E, 1 seed)", "both", "A_full_fitness_set+E_full_fitness_set", 1, ROUNDS, 1.0),
    ("T1-bounds", 1, "3*", "centralized + local-only bounds", "both", "centralized+local_only", 3, ROUNDS, 1.0),
    ("OPT-paper", 0, "opt", "FedGA (paper GA settings): P=10, G=20, 20 rounds, batch 8, 1 seed", "both", "fedga_paper_ga_settings", 1, 20, 1.0),
    ("T2-fw-step0", 2, "6", "fitness-weight Step 0: 1 run with saved candidate pools (offline re-rank)", "both", "E_xfedga", 1, ROUNDS, 1.0),
    ("T2-fw-corners", 2, "6", "fitness-weight corners + 3 extra points (breast), transfer check (prostate)", "both", "E_xfedga", 6 * 3, ROUNDS, 1.0),
    ("T2-sens", 2, "6", "sensitivity: pop {4,10}, gen {3,10}, sigma0 {0.005,0.02}, tau {1,2,5} (breast)", "breast", "E_xfedga", 9 * 3, ROUNDS, 1.0),
    ("T2-hetero", 2, "7", "heterogeneous prostate partition: FedAvg, A, E", "prostate", "fedavg+A_fedga+E_xfedga", 5, ROUNDS, 1.0),
    ("T2-robust", 2, "9", "robustness: q_i=1 client, corrupted-label client; E and FedAvg (breast)", "breast", "E_xfedga+fedavg", 2 * 3, ROUNDS, 1.0),
    ("T2-dp", 2, "9", "DP: clipping + Gaussian noise, XFedGA (breast, optional)", "breast", "E_xfedga", 3, ROUNDS, 1.0),
]
TEST_IMAGES = {"prostate": 242, "breast": 300}   # explanation metrics: all prostate test slices (~16 cases), fixed 300-image breast subsample
EXPL_CHECKPOINTS = 4 * 5   # FedAvg, FedProx, A, E x 5 seeds, per task


def budget(start, end, build_until, wd_h=8, we_h=16):
    tot, after = 0, 0
    d = start
    while d <= end:
        h = we_h if d.weekday() >= 5 else wd_h
        tot += h
        if d > build_until:
            after += h
        d += dt.timedelta(days=1)
    return tot, after


def guide_review_plan(comp):
    """Revised Tier 1 for the Guide Review on 30 Oct 2026 (approved 15 Oct). Current timings, no speed-up assumed."""
    both = ["prostate", "breast"]

    def h(methods, tasks, n, rounds=ROUNDS, factor=1.0):
        return sum(run_hours(comp[t], m, t, rounds, factor) for t in tasks for m in methods) * n

    items = [
        ("S2-remeasure", "Session 2", "re-measure after speed-up: FedGA + XFedGA x 3 rounds per task, + sharing re-test",
         h(["A_fedga", "E_xfedga"], both, 1, rounds=3) + h(["A_fedga"], ["prostate"], 2, rounds=2)),
        ("S3-flsmoke", "Session 3", "fl_dist 1-round smoke per task + memory measurement",
         h(["E_xfedga"], both, 1, rounds=1, factor=1.15)),
        ("S4-core-3seeds", "Session 4", "FedAvg, FedGA (baseline), XFedGA (beta=0.5) x 3 seeds x 2 tasks",
         h(["fedavg", "A_fedga", "E_xfedga"], both, 3)),
        ("S4-beta-screen", "Session 4", "beta screen {0, 1.0} x 1 seed x 2 tasks (beta=0.5 reused from core seed 1)",
         h(["E_xfedga"], both, 2)),
        ("S4-parity-1seed", "Session 4", "fl_dist XFedGA, breast, 1 seed (custom seed reused)",
         h(["E_xfedga"], ["breast"], 1, factor=1.15)),
        ("S4-parity-extra2", "Session 4 (if time)", "parity seeds 2-3",
         h(["E_xfedga"], ["breast"], 2, factor=1.15)),
        ("S4-core-seeds45", "Session 4 (if time)", "core seeds 4-5",
         h(["fedavg", "A_fedga", "E_xfedga"], both, 2)),
        ("S5-test-eval", "Session 5", "test evaluation + Grad-CAM overlays: 3 methods x 3 seeds x 2 tasks (setup + 1 eval each)",
         sum((comp[t]["S"] + 3 * comp[t]["V"]) / 3600 for t in both) * 9),
    ]
    rows = [{"id": i, "session": s, "label": l, "raw_h": v, "with_contingency_h": v * CONTINGENCY}
            for i, s, l, v in items]

    def window(a, b):
        return budget(a, b, a - dt.timedelta(days=1), *BUDGET)[0]
    windows = {
        "2026-10-16..17 (Session 2 build; GPU for smoke/re-measure only)": window(dt.date(2026, 10, 16), dt.date(2026, 10, 17)),
        "2026-10-18..21 (Session 3 build; GPU free for core runs if Session 2 is done)": window(dt.date(2026, 10, 18), dt.date(2026, 10, 21)),
        "2026-10-22..26 (Session 4)": window(dt.date(2026, 10, 22), dt.date(2026, 10, 26)),
        "2026-10-27..29 (Session 5)": window(dt.date(2026, 10, 27), dt.date(2026, 10, 29)),
    }
    must = [r for r in rows if "if time" not in r["session"]]
    return {"rows": rows, "windows_gpu_h": windows,
            "must_total_raw_h": sum(r["raw_h"] for r in must),
            "must_total_with_contingency_h": sum(r["with_contingency_h"] for r in must),
            "if_time_total_with_contingency_h": sum(r["with_contingency_h"] for r in rows if "if time" in r["session"]),
            "session4_must_with_contingency_h": sum(r["with_contingency_h"] for r in must if r["session"] == "Session 4")}


RUNNER_DIR = os.path.join(REPO, "results", "timing", "runner")


def load_runner_timings():
    """
    Per (task, method): 15-round run hours EXTRAPOLATED from 3-round runs of the new runner
    (experiments/run_federated.py, tag timing3): setup + 15 x mean round + measured test evaluation.
    These runs use the final Session-2 protocol (cache, D-1, three-way splits, shared 40-image
    fitness subsample, elite cache, per-round validation, deterministic kernels).
    """
    import csv
    import glob
    out = {}
    for meta_path in glob.glob(os.path.join(RUNNER_DIR, "*_timing3", "run_meta.json")):
        meta = json.load(open(meta_path, encoding="utf-8"))
        rows = list(csv.DictReader(open(os.path.join(os.path.dirname(meta_path), "round_log.csv"), encoding="utf-8")))
        rounds = [float(r["round_s"]) for r in rows]
        test_s = meta["total_s"] - meta["setup_s"] - sum(rounds)
        out[(meta["task"], meta["method"])] = {
            "rounds_measured": len(rounds), "mean_round_s": sum(rounds) / len(rounds), "round_s": rounds,
            "setup_s": meta["setup_s"], "test_and_selection_s": test_s,
            "fitness_evals_per_round": [int(r["fitness_evals"]) for r in rows],
            "peak_reserved_MiB": max(int(r["peak_reserved_MiB"]) for r in rows),
            "hours_15_rounds": (meta["setup_s"] + ROUNDS * sum(rounds) / len(rounds) + test_s) / 3600,
            "extrapolated": True, "meta_path": meta_path,
            "n_test": meta["server_sizes"]["test"], "n_val": meta["server_sizes"]["validation"]}
    return out


def gpu_state_summary(path=os.path.join(RUNNER_DIR, "gpu_state.csv")):
    """nvidia-smi samples logged during the runner timings: clock/power while busy, throttle share."""
    import csv
    if not os.path.exists(path):
        return None
    busy = []
    for r in csv.DictReader(open(path, encoding="utf-8", errors="ignore"), skipinitialspace=True):
        try:
            util = float(r["utilization.gpu [%]"].split()[0])
            clk = float(r["clocks.current.sm [MHz]"].split()[0])
            pwr = float(r["power.draw [W]"].split()[0])
            reason = int(r["clocks_event_reasons.active"], 16)
        except (KeyError, ValueError, AttributeError):
            continue
        if util >= 50:
            busy.append((clk, pwr, reason))
    if not busy:
        return None
    clocks = sorted(b[0] for b in busy)
    powers = sorted(b[1] for b in busy)
    throttled = sum(1 for b in busy if b[2] & ~0x1)   # any reason other than "GPU idle"
    return {"busy_samples": len(busy), "median_sm_clock_MHz": clocks[len(clocks) // 2],
            "min_sm_clock_MHz": clocks[0], "median_power_W": powers[len(powers) // 2],
            "share_busy_samples_with_throttle_flag": throttled / len(busy)}


def guide_review_plan_runner(rt):
    """Guide Review plan costed from the new-runner measurements (supersedes guide_review_plan)."""
    both = ["prostate", "breast"]
    methods = ["fedavg", "fedga", "xfedga"]

    def h(meths, tasks, n, factor=1.0):
        return sum(rt[(t, m)]["hours_15_rounds"] for t in tasks for m in meths) * n * factor

    def h1(meth, task, factor=1.0):  # 1-round smoke, no test evaluation
        r = rt[(task, meth)]
        return (r["setup_s"] + r["round_s"][0]) / 3600 * factor
    items = [
        ("S3-flsmoke", "Session 3", "fl_dist 1-round smoke per task + memory measurement",
         sum(h1("xfedga", t, 1.15) for t in both)),
        ("S4-core-5seeds", "Session 4", "FedAvg, FedGA (baseline), XFedGA (beta=0.5) x 5 seeds x 2 tasks (headline)",
         h(methods, both, 5)),
        ("S4-beta-screen", "Session 4", "beta screen {0, 1.0} x 1 seed x 2 tasks (beta=0.5 reused from core seed 1)",
         h(["xfedga"], both, 2)),
        ("S4-parity-1seed", "Session 4", "fl_dist XFedGA, breast, 1 seed (custom seed reused)",
         h(["xfedga"], ["breast"], 1, 1.15)),
        ("S4-parity-extra2", "Session 4 (if time)", "parity seeds 2-3", h(["xfedga"], ["breast"], 2, 1.15)),
        ("S5-overlays", "Session 5", "Grad-CAM overlays from saved checkpoints (test already evaluated in-run)", 0.25),
    ]
    rows = [{"id": i, "session": s, "label": l, "raw_h": v, "with_contingency_h": v * CONTINGENCY}
            for i, s, l, v in items]
    must = [r for r in rows if "if time" not in r["session"]]
    return {"source": "results/timing/runner/*_timing3 (3 rounds, extrapolated to 15)", "per_run": {
                f"{t}/{m}": v for (t, m), v in sorted(rt.items())},
            "rows": rows,
            "session4_must_with_contingency_h": sum(r["with_contingency_h"] for r in must if r["session"] == "Session 4"),
            "if_time_total_with_contingency_h": sum(r["with_contingency_h"] for r in rows if "if time" in r["session"]),
            "all_session4_with_contingency_h": sum(r["with_contingency_h"] for r in rows if r["session"].startswith("Session 4"))}


FULL_CONTINGENCY = 1.2   # resumable overnight queue: small packing loss; reruns and smokes
FULL_BUDGET = (8, 16)    # GPU-h per weekday night / weekend day (instruction 15 Oct, project deadline 7 Nov)
FULL_WINDOW = (dt.date(2026, 10, 18), dt.date(2026, 11, 5))


def newcam_factors(rt, task):
    import csv
    new = glob.glob(os.path.join(REPO, "results", "smoke", f"{task}_xfedga_seed1_*_newcam_timing", "round_log.csv"))
    if not new:
        return 1.0, 1.0
    n = next(csv.DictReader(open(new[0], encoding="utf-8")))
    o = next(csv.DictReader(open(os.path.join(os.path.dirname(rt[(task, "xfedga")]["meta_path"]), "round_log.csv"),
                                 encoding="utf-8")))
    return float(n["qi_s"]) / float(o["qi_s"]), float(n["ga_s"]) / float(o["ga_s"])


def method_round_costs(rt, task):
    """Per-round seconds for every method, from the measured runner logs of FedAvg / FedGA / XFedGA."""
    import csv

    def mean_col(method, col):
        d = os.path.dirname(rt[(task, method)]["meta_path"])
        rows = list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
        return sum(float(r[col]) for r in rows) / len(rows)
    L = mean_col("fedavg", "local_s")
    V = sum(mean_col(m, "val_s") for m in ["fedavg", "fedga", "xfedga"]) / 3
    Q = mean_col("xfedga", "qi_s")
    G_dsc, G_multi = mean_col("fedga", "ga_s"), mean_col("xfedga", "ga_s")
    # new CAM definition (17 Oct): Grad-CAM-based costs (q_i, multi-objective GA) scaled by the measured
    # round-1 ratio new/old (results/smoke/<task>_xfedga_seed1_*_newcam_timing vs the timing3 run)
    fq, fg = newcam_factors(rt, task)
    Q, G_multi = Q * fq, G_multi * fg
    xfedga_round = rt[(task, "xfedga")]["mean_round_s"] + (fq - 1) * mean_col("xfedga", "qi_s")         + (fg - 1) * mean_col("xfedga", "ga_s")
    return {
        "fedavg": rt[(task, "fedavg")]["mean_round_s"], "fedga": rt[(task, "fedga")]["mean_round_s"],
        "xfedga": xfedga_round, "newcam_factors": {"qi": fq, "ga_multi": fg},
        "B_weighted_only": L + Q + V, "C_fedavg_ga_multi": L + G_multi + V, "D_fedga_anneal": L + G_dsc + V,
        "F_random_qi": L + G_multi + V, "random_search": L + G_multi + V,
        "fedprox": 1.1 * L + V, "centralized": L + V, "local_only": L + 2 * V,
        "components": {"local": L, "qi": Q, "ga_dsc": G_dsc, "ga_multi": G_multi, "val": V},
    }


def full_plan(rt):
    """Whole-project GPU queue (Phase 1 + 2), costed from the re-timed runner; calendar fill."""
    both = ["prostate", "breast"]
    rc = {t: method_round_costs(rt, t) for t in both}

    def run_h(task, method, factor=1.0):
        base = rt[(task, "xfedga" if method not in rt_methods else method)]
        return factor * (base["setup_s"] + ROUNDS * rc[task][method] + base["test_and_selection_s"]) / 3600
    rt_methods = {"fedavg", "fedga", "xfedga"}

    def runs(methods, tasks, seeds, factor=1.0):
        return sum(run_h(t, m, factor) for t in tasks for m in methods) * seeds

    def paper_settings_h(task):
        # elite cache on: P + G*P/2 = 10 + 20*5 = 110 evaluations per round (vs 21 measured), each on the
        # full fitness set instead of the 40-image subsample (DSC fitness scales with image count)
        r = rt[(task, "fedga")]
        c = rc[task]["components"]
        meta = json.load(open(r["meta_path"], encoding="utf-8"))
        ga = c["ga_dsc"] * (110 / 21) * (meta["server_sizes"]["fitness"] / 40)
        return (r["setup_s"] + 20 * (c["local"] + ga + c["val"]) + r["test_and_selection_s"]) / 3600

    def xai_h(task, n_ckpt, n_images):
        # per image: 12 CAM-equivalents (2 CAM methods + 10-layer randomisation) + 42 forwards (deletion+insertion)
        r = rt[(task, "xfedga")]
        cam = r["test_and_selection_s"] / r["n_test"]               # measured: 1 CAM + metrics per test image
        fwd = rc[task]["components"]["val"] / r["n_val"]            # measured: forward + DSC/HD95 per val image
        return n_ckpt * n_images * (12 * cam + 42 * fwd) / 3600
    queue = [
        ("headline", "FedAvg, FedGA (baseline), XFedGA x 2 tasks x 5 seeds", runs(["fedavg", "fedga", "xfedga"], both, 5), "never"),
        ("beta_screen", "XFedGA beta in {0, 1.0} x 2 tasks x 3 seeds (beta 0.5 = headline seeds 1-3)", runs(["xfedga"], both, 2 * 3), "keep"),
        ("xai_metrics", "independent XAI metrics on 30 headline checkpoints (prostate: all test slices; breast: fixed 300-image subsample)",
         xai_h("prostate", 15, rt[("prostate", "xfedga")]["n_test"]) + xai_h("breast", 15, 300), "never"),
        ("ablations_BF_RS", "ablations B, F + random search x 2 tasks x 3 seeds (A, E reused from headline)",
         runs(["B_weighted_only", "F_random_qi", "random_search"], both, 3), "never"),
        ("ablations_CD", "ablations C, D x 2 tasks x 3 seeds", runs(["C_fedavg_ga_multi", "D_fedga_anneal"], both, 3), "cut4"),
        ("parity", "fl_dist XFedGA, breast, 3 seeds (custom-loop seeds reused)", runs(["xfedga"], ["breast"], 3, 1.15), "never"),
        ("fedprox", "FedProx x 2 tasks x 5 seeds", runs(["fedprox"], both, 5), "keep"),
        ("local_only", "local-only bound x 2 tasks x 3 seeds", runs(["local_only"], both, 3), "keep"),
        ("centralized", "centralized bound x 2 tasks x 3 seeds", runs(["centralized"], both, 3), "cut3"),
        ("hetero", "heterogeneous prostate partition: FedAvg, FedGA, XFedGA x 3 seeds", runs(["fedavg", "fedga", "xfedga"], ["prostate"], 3), "keep"),
        ("robustness", "noisy-q_i client (XFedGA) and corrupted-label client (XFedGA + FedAvg), breast, 3 seeds",
         runs(["xfedga"], ["breast"], 3) + runs(["xfedga", "fedavg"], ["breast"], 3), "keep"),
        ("fw_grid", "fitness-weight grid (only if offline Step 0 shows the winner changes): 4 points x 3 seeds breast + 2 x 3 prostate",
         runs(["xfedga"], ["breast"], 12) + runs(["xfedga"], ["prostate"], 6), "cut2"),
        ("dp", "DP (clip + Gaussian noise), XFedGA, breast, 3 seeds", runs(["xfedga"], ["breast"], 3), "cut1"),
        ("paper_ga_settings", "optional: FedGA (paper GA settings) P=10, G=20, 20 rounds, full fitness set, batch 8, 1 seed per task",
         sum(paper_settings_h(t) for t in both), "cut0"),
    ]
    days, d = [], FULL_WINDOW[0]
    while d <= FULL_WINDOW[1]:
        days.append((d, FULL_BUDGET[1] if d.weekday() >= 5 else FULL_BUDGET[0]))
        d += dt.timedelta(days=1)
    capacity = sum(h for _, h in days)

    def finish_dates(items):
        out, used = [], 0.0
        for name, label, h, cut in items:
            used += h * FULL_CONTINGENCY
            acc, when = 0.0, None
            for day, cap in days:
                acc += cap
                if acc >= used:
                    when = day
                    break
            out.append({"item": name, "label": label, "raw_h": h, "with_contingency_h": h * FULL_CONTINGENCY,
                        "cumulative_h": used, "finishes": str(when) if when else "after window", "cut_rank": cut})
        return out
    rows = finish_dates(queue)
    total = sum(r["with_contingency_h"] for r in rows)
    cuts, kept = [], list(queue)
    for rank in ["cut0", "cut1", "cut2", "cut3", "cut4"]:
        if sum(q[2] for q in kept) * FULL_CONTINGENCY <= capacity:
            break
        cuts.append(rank)
        kept = [q for q in kept if q[3] != rank]
    return {"window": f"{FULL_WINDOW[0]}..{FULL_WINDOW[1]}", "budget": f"{FULL_BUDGET[0]} h/weekday night, {FULL_BUDGET[1]} h/weekend day",
            "capacity_h": capacity, "contingency": FULL_CONTINGENCY, "round_costs_s": rc,
            "queue": rows, "total_with_contingency_h": total, "fits": total <= capacity,
            "cuts_needed": cuts, "queue_after_cuts": finish_dates(kept) if cuts else None}


# Schedule approved 17 Oct 2026: full GPU days and "some evenings/nights" (assumed 6 GPU-h; flagged).
DAY_HOURS_17OCT = {dt.date(2026, 10, 17): 10,   # from 14:00 (12 h from noon minus ~2 h of the stopped old-CAM screen)
                   dt.date(2026, 10, 20): 16, dt.date(2026, 10, 21): 16, dt.date(2026, 10, 23): 12,
                   dt.date(2026, 10, 24): 16, dt.date(2026, 10, 25): 16}
EVENING_HOURS = 6
WINDOW_17OCT = (dt.date(2026, 10, 17), dt.date(2026, 11, 5))
BUFFER_MIN_H = 60


def plan_17oct(rt):
    """Queue approved 17 Oct: beta screen -> headline (5 seeds) -> ablations + random search (5 seeds) ->
    parity -> XAI metrics -> FedProx + bounds -> heterogeneous (5 seeds) -> robustness / DP (3 seeds)."""
    both = ["prostate", "breast"]
    rc = {t: method_round_costs(rt, t) for t in both}
    measured = {"fedavg", "fedga", "xfedga"}

    def run_h(task, method, factor=1.0, test=True):
        base = rt[(task, method if method in measured else "xfedga")]
        test_s = base["test_and_selection_s"] if test else 0.0
        return factor * (base["setup_s"] + ROUNDS * rc[task][method] + test_s) / 3600

    def runs(methods, tasks, seeds, factor=1.0, test=True):
        return sum(run_h(t, m, factor, test) for t in tasks for m in methods) * seeds

    def xai_h(task, n_ckpt, n_images):
        r = rt[(task, "xfedga")]
        cam = r["test_and_selection_s"] / r["n_test"]
        fwd = rc[task]["components"]["val"] / r["n_val"]
        return n_ckpt * n_images * (12 * cam + 42 * fwd) / 3600
    queue = [
        ("beta_screen", "XFedGA beta {0, 0.2, 0.5, 1.0} x 2 tasks x 2 seeds, validation only (new CAM; rerun from 17 Oct 14:00)",
         runs(["xfedga"], both, 4 * 2, test=False)),
        ("headline", "FedAvg, FedGA (baseline), XFedGA (selected beta) x 2 tasks x 5 seeds",
         runs(["fedavg", "fedga", "xfedga"], both, 5)),
        ("ablations", "ablations B, C, D, F + random search x 2 tasks x 5 seeds (A = FedGA, E = XFedGA from headline)",
         runs(["B_weighted_only", "C_fedavg_ga_multi", "D_fedga_anneal", "F_random_qi", "random_search"], both, 5)),
        ("parity", "fl_dist XFedGA, breast, 3 seeds", runs(["xfedga"], ["breast"], 3, 1.15)),
        ("xai_metrics", "independent XAI metrics on the 30 headline checkpoints (no retraining)",
         xai_h("prostate", 15, rt[("prostate", "xfedga")]["n_test"]) + xai_h("breast", 15, 300)),
        ("fedprox_bounds", "FedProx x 2 tasks x 5 seeds; centralized + local-only bounds x 2 tasks x 3 seeds",
         runs(["fedprox"], both, 5) + runs(["centralized", "local_only"], both, 3)),
        ("hetero", "heterogeneous prostate partition: FedAvg, FedGA, XFedGA x 5 seeds",
         runs(["fedavg", "fedga", "xfedga"], ["prostate"], 5)),
        ("robustness_dp", "breast, 3 seeds: noisy-q_i client (XFedGA), corrupted-label client (XFedGA + FedAvg), DP (XFedGA)",
         runs(["xfedga"], ["breast"], 3) + runs(["xfedga", "fedavg"], ["breast"], 3) + runs(["xfedga"], ["breast"], 3)),
    ]
    days, d = [], WINDOW_17OCT[0]
    while d <= WINDOW_17OCT[1]:
        days.append((d, DAY_HOURS_17OCT.get(d, EVENING_HOURS)))
        d += dt.timedelta(days=1)
    capacity = sum(h for _, h in days)
    rows, used = [], 0.0
    for name, label, h in queue:
        used += h * FULL_CONTINGENCY
        acc, when = 0.0, "after window"
        for day, cap in days:
            acc += cap
            if acc >= used:
                when = str(day)
                break
        rows.append({"item": name, "label": label, "raw_h": h, "with_contingency_h": h * FULL_CONTINGENCY,
                     "cumulative_h": used, "finishes": when})
    by_guide = sum(cap for day, cap in days if day <= dt.date(2026, 10, 29))
    return {"window": f"{WINDOW_17OCT[0]}..{WINDOW_17OCT[1]}", "capacity_h": capacity,
            "full_days": {str(k): v for k, v in DAY_HOURS_17OCT.items()}, "evening_hours_assumed": EVENING_HOURS,
            "contingency": FULL_CONTINGENCY, "queue": rows, "planned_with_contingency_h": used,
            "unallocated_buffer_h": capacity - used, "buffer_ok": capacity - used >= BUFFER_MIN_H,
            "capacity_by_2026_10_29_h": by_guide, "finish": rows[-1]["finishes"]}


def main():
    comp, sharing = load_timing()
    out = {"components": comp, "gpu_sharing": sharing, "assumptions": ASSUMPTIONS, "experiments": []}
    for (eid, tier, prio, label, tasks, methods, seeds, rounds, factor) in EXPERIMENTS:
        task_list = ["prostate", "breast"] if tasks == "both" else [tasks]
        per_task = {}
        for t in task_list:
            if comp[t] is None:
                per_task[t] = None
                continue
            per_run = sum(run_hours(comp[t], m, t, rounds, factor) for m in methods.split("+"))
            per_task[t] = {"hours_per_run": per_run, "runs": seeds, "total_h": per_run * seeds}
        total = sum(v["total_h"] for v in per_task.values() if v) if all(per_task.values()) else None
        out["experiments"].append({"id": eid, "tier": tier, "priority": prio, "label": label,
                                   "methods": methods, "seeds_or_runs": seeds, "rounds": rounds,
                                   "factor": factor, "per_task": per_task, "total_h": total})
    if all(comp.values()):
        expl = {t: EXPL_CHECKPOINTS * expl_eval_hours(comp[t], TEST_IMAGES[t]) for t in comp}
        out["experiments"].append({"id": "T2-expl", "tier": 2, "priority": "8",
                                   "label": "NSD + independent explanation metrics (eval only, 20 checkpoints/task)",
                                   "methods": "eval", "seeds_or_runs": EXPL_CHECKPOINTS, "rounds": 0, "factor": 1.0,
                                   "per_task": {t: {"hours_per_run": expl[t] / EXPL_CHECKPOINTS,
                                                    "runs": EXPL_CHECKPOINTS, "total_h": expl[t]} for t in expl},
                                   "total_h": sum(expl.values())})
        sums = {}
        for e in out["experiments"]:
            key = {1: "tier1", 2: "tier2", 0: "optional"}[e["tier"]]
            core = e["priority"] in ("1b", "2", "3", "4", "5")
            sums.setdefault(key, 0.0)
            sums[key] += e["total_h"]
            if e["tier"] == 1:
                k2 = "tier1_required" if core else "tier1_cuttable"
                sums[k2] = sums.get(k2, 0.0) + e["total_h"]
        out["totals_h"] = sums
        beta_e = next(e for e in out["experiments"] if e["id"] == "T1-beta")
        out["beta_alternatives"] = {
            "full_grid_5beta_3seeds_15r": beta_e["total_h"],
            "reduced_grid_3beta_3seeds_15r": beta_e["total_h"] * 3 / 5,
            "screen_5beta_1seed_8r_then_top2_x3seeds_15r": sum(
                run_hours(comp[t], "E_xfedga", t, 8) * 5 + run_hours(comp[t], "E_xfedga", t, 15) * 2 * 3 for t in comp),
        }
    start, end = dt.date(2026, 10, 15), dt.date(2026, 11, 8)
    out["budget"] = {"window": f"{start}..{end}", "assumption": "scenarios below; confirmed budget = 4 h per weekday, 10 h per weekend day (15 Oct)"}
    for label, build_until in [("no_build_gap", start - dt.timedelta(days=1)),
                               ("build_until_2026-10-22", dt.date(2026, 10, 22))]:
        for wd_h, we_h in [(8, 16), (4, 10), (0, 12)]:
            tot, after = budget(start, end, build_until, wd_h, we_h)
            out["budget"][f"{label}__{wd_h}h_weekday_{we_h}h_weekend"] = {
                "nominal_h": tot, "available_after_build_h": after}
    out["contingency_factor"] = CONTINGENCY
    if "totals_h" in out:
        out["totals_with_contingency_h"] = {k: v * CONTINGENCY for k, v in out["totals_h"].items()}
    if sharing:
        out["budget"]["effective_multiplier_if_2_parallel"] = sharing["throughput_factor"]
    if "totals_h" in out:
        cap = out["budget"][f"build_until_2026-10-22__{BUDGET[0]}h_weekday_{BUDGET[1]}h_weekend"]["available_after_build_h"]
        by_id = {e["id"]: e["total_h"] for e in out["experiments"]}
        sched, cum = [], {s: 0.0 for s in SLOWDOWNS}
        for i, eid in enumerate(FILL_ORDER, 1):
            row = {"order": i, "id": eid, "raw_h": by_id[eid], "with_contingency_h": by_id[eid] * CONTINGENCY}
            for s in SLOWDOWNS:
                cum[s] += by_id[eid] * CONTINGENCY * s
                row[f"cum_h_at_{s}x"] = cum[s]
                row[f"fits_at_{s}x"] = cum[s] <= cap
            sched.append(row)
        out["schedule"] = {"budget_h": cap, "budget": f"{BUDGET[0]} h/weekday, {BUDGET[1]} h/weekend day, 2026-10-23..2026-11-08 (confirmed 15 Oct)",
                           "slowdowns": SLOWDOWNS, "rows": sched}
    if all(comp.values()):
        out["guide_review_pre_step0"] = guide_review_plan(comp)
    rt = load_runner_timings()
    if len(rt) == 6:
        out["guide_review"] = guide_review_plan_runner(rt)
        out["guide_review"]["gpu_state_during_timing"] = gpu_state_summary()
        out["full_plan"] = full_plan(rt)
        out["plan_17oct"] = plan_17oct(rt)
        start = dt.date(2026, 10, 22)
        out["guide_review"]["session4_window_gpu_h"] = budget(start, dt.date(2026, 10, 26), start - dt.timedelta(days=1), *BUDGET)[0]
    json.dump(out, open(os.path.join(HERE, "plan_costs.json"), "w", encoding="utf-8"), indent=2, default=str)

    lines = ["| id | tier | prio | experiment | per-run h (prostate / breast) | runs per task | total GPU-h |", "|---|---|---|---|---|---|---|"]
    for e in out["experiments"]:
        pt = e["per_task"]
        pr = " / ".join(f"{pt[t]['hours_per_run']:.2f}" if pt.get(t) else "–" for t in ["prostate", "breast"])
        tot = f"{e['total_h']:.1f}" if e["total_h"] is not None else "pending"
        lines.append(f"| {e['id']} | {e['tier'] or 'opt'} | {e['priority']} | {e['label']} | {pr} | {e['seeds_or_runs']} | {tot} |")
    open(os.path.join(HERE, "plan_costs.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({k: out.get(k) for k in ["totals_h", "totals_with_contingency_h", "beta_alternatives", "budget"]}, indent=2))


if __name__ == "__main__":
    main()
