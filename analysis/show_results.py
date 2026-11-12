"""
Print the final results as plain-text tables in the terminal. Read-only: no training, no GPU, no file is written
(analysis outputs are computed in memory or in a temporary folder that is deleted afterwards).

Every number comes from the same analysis functions that produced the report tables, so the values match:
  1  Review 2 re-check: old (legacy) FedGA vs old XFedGA checkpoints, corrected evaluation
       <- results/diagnostics/legacy_reeval.json (analysis/legacy_reeval.py; the legacy checkpoints are not run
          folders, so this section reads that saved evaluation instead of results/runs/)
  2  main 5-seed comparison, breast + prostate_ms   <- analysis/make_tables.py (--group main,final --comparisons final)
  3  RUNMC independent hospital + breast seeds 6-10 <- analysis/g_site_runmc.py, analysis/g2_replication.py
  4  corrupted-label dose-response + dishonest client (q_i = 1)
                                                    <- analysis/dose_response.py, analysis/post_items_analysis.py (d)
  5  ablations, fitness-weight screen, beta screen  <- post_items_analysis.py (e), sensitivity_screens.py,
                                                       fitness_weight_step0.py, beta_decision.py
  6  paper-GA FedGA, FedProx, centralised / local-only bounds  <- post_items_analysis.py (c, f)
  7  Flower vs single-process parity                <- post_items_analysis.py (b)

Usage (PowerShell):
    & $py analysis/show_results.py 2
    & $py analysis/show_results.py all
    & $py analysis/show_results.py 2 --all-metrics      # section 2: every explanation metric, not only the
                                                        # pre-registered ones
"""
import sys

sys.dont_write_bytecode = True      # read-only: do not even write analysis/__pycache__

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

# scipy's Wilcoxon (inside stats.compare) warns about small samples / zero differences; the printed tables do not
# use that p-value, and the warnings clutter the console.
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import beta_decision  # noqa: E402
import dose_response  # noqa: E402
import fitness_weight_step0 as step0  # noqa: E402
import g2_replication  # noqa: E402
import g_site_runmc  # noqa: E402
import make_tables  # noqa: E402
import post_items_analysis as post  # noqa: E402
import sensitivity_screens  # noqa: E402
from stats import outcome  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_GLOB = os.path.join(REPO, "results", "runs", "*")
SHORT = {"fedavg": "FedAvg", "fedga": "FedGA", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+",
         "xfedga_plus_dsc": "XFedGA+ (DSC-only)", "fedprox": "FedProx", "fedga_paper": "FedGA (paper GA)",
         "abl_B": "B: q_i weighting only", "abl_C": "C: FedAvg + GA (multi)", "abl_D": "D: FedGA + annealing",
         "abl_F": "F: XFedGA, random q_i", "random_search": "random search", "centralized": "centralised",
         "local_only": "local-only"}
METRIC = {"dsc": "DSC", "hd95_norm": "HD95n", "ec": "EC", "pg": "PG", "ec_mass": "EC_mass",
          "pg_gradcampp": "PG (Grad-CAM++)", "rho_after_all_layers": "rand. rho", "pg_gradcam": "PG (Grad-CAM)",
          "ec_iou_gradcam": "EC (IoU)", "ec_mass_gradcam": "EC_mass", "deletion_auc": "deletion AUC",
          "insertion_auc": "insertion AUC"}
ASCII = str.maketrans({"±": "+/-", "−": "-", "Δ": "d", "–": "-", "—": "-", "β": "beta"})


# ---------------------------------------------------------------- formatting
def pm(m, s, nd=4):
    if not np.isfinite(m):
        return "-"
    return f"{m:.{nd}f} +/- {s:.{nd}f}" if np.isfinite(s) else f"{m:.{nd}f}"


def sgn(x, nd=4):
    return f"{x:+.{nd}f}" if np.isfinite(x) else "-"


def ci(lo, hi, nd=4):
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def table(header, rows, left=1, note=None):
    """Print a fixed-width table; the first `left` columns are left-aligned, the rest right-aligned."""
    cells = [[str(c).translate(ASCII) for c in r] for r in [header] + rows]
    w = [max(len(r[i]) for r in cells) for i in range(len(header))]

    def line(r):
        return "  " + "  ".join(c.ljust(w[i]) if i < left else c.rjust(w[i]) for i, c in enumerate(r))
    print(line(cells[0]))
    print("  " + "  ".join("-" * x for x in w))
    for r in cells[1:]:
        print(line(r))
    if note:
        print(wrap(note))
    print()


def wrap(s):
    """Word-wrap a note to fit a 120-column PowerShell window."""
    return textwrap.fill(s.translate(ASCII), width=118, initial_indent="  ", subsequent_indent="  ")


def subtitle(s):
    print(wrap(s))
    print()


def title(n, s):
    bar = "=" * 100
    print(bar)
    print(f" SECTION {n}  {s}".translate(ASCII))
    print(bar)
    print()


# ---------------------------------------------------------------- reuse helpers
class _Sink(io.StringIO):
    """Swallows a script's own printout (the scripts call sys.stdout.reconfigure)."""
    def reconfigure(self, **_):
        pass


def main_into_tempdir(module, json_name):
    """Run an analysis script's main() with its OUT folder redirected to a temporary folder; return its JSON.
    Same code path as the report, and no repository file is written."""
    old = module.OUT
    with tempfile.TemporaryDirectory() as tmp:
        module.OUT = tmp
        try:
            with contextlib.redirect_stdout(_Sink()):
                module.main()
            with open(os.path.join(tmp, json_name), encoding="utf-8") as f:
                return json.load(f)
        finally:
            module.OUT = old


@contextlib.contextmanager
def track_post_runs(used):
    """Record every run folder that post_items_analysis.run_dir resolves (to count the runs a section used)."""
    orig = post.run_dir

    def wrapped(*a, **k):
        d = orig(*a, **k)
        used.add(os.path.basename(d))
        return d
    post.run_dir = wrapped
    try:
        yield
    finally:
        post.run_dir = orig


def arm_dsc(task, method, seeds, tag=None):
    """Test DSC mean +/- s.d. over seeds (per-run value = mean over test cases, as the headline summary table)."""
    vals = [make_tables.run_value(rows, "dsc") for rows in post.runs(task, method, seeds, tag)]
    return make_tables.mean_sd_over_seeds(vals)


def cmp_rows(rows):
    """post_items comparison dicts -> table rows."""
    return [[r["task"], r["comparison"].replace(" (baseline)", ""), r["n"], sgn(r["diff"]), ci(r["lo"], r["hi"]),
             r["outcome"], r["t_outcome"]] for r in rows]


CMP_HEADER = ["task", "comparison", "cases", "d test DSC", "95% CI", "outcome", "t-interval"]


# ---------------------------------------------------------------- sections
def section1(_args):
    title(1, "Review 2 re-check: old FedGA vs old XFedGA (legacy breast checkpoints, corrected evaluation)")
    path = os.path.join(REPO, "results", "diagnostics", "legacy_reeval.json")
    d = json.load(open(path, encoding="utf-8"))
    rows = []
    for k in ["dsc", "hd95_norm", "ec", "ec_mass", "pg"]:
        v = d["metrics"][k]
        c = v["corrected"]
        surv = "-" if v["claim_survives"] is None else ("yes" if v["claim_survives"] else "no")
        rows.append([METRIC[k], f"{v['mean']['fedga']:.4f}", f"{v['mean']['xfedga']:.4f}", sgn(c["mean_diff"]),
                     ci(c["ci_low"], c["ci_high"]), c["outcome"], v["review2_claim"] or "(not on slide)", surv])
    table(["metric", "old FedGA", "old XFedGA", "d (X - F)", "95% CI (patients)", "outcome", "Review 2 claim",
           "survives"], rows,
          note="One final-round checkpoint per method (single unseeded runs), so there is no s.d. over seeds.")
    subtitle(f"Test images: {d['n_images']} from {d['n_patients']} BUS-BRA patients ({d['excluded_patients_in_old_fitness_half']} "
             f"of {d['test_patients_total']} test patients excluded: they were in the old fitness half).")
    subtitle("Source: results/diagnostics/legacy_reeval.json (analysis/legacy_reeval.py). The legacy checkpoints live in "
             "results/legacy/breast/, not in results/runs/.")
    return 0


def section2(args):
    title(2, "Main 5-seed comparison (FedAvg, FedGA, XFedGA, XFedGA+, XFedGA+ DSC-only; breast, prostate_ms)")
    # exactly as: make_tables.py --group main,final --comparisons final --tasks breast,prostate_ms
    runs = {r["run_id"]: r for g in ["main", "final"] for r in make_tables.load_runs(RUNS_GLOB, g)}
    pairs = make_tables.COMPARISONS["final"]
    in_comp = {m for p in pairs for m in p}
    runs = [r for r in runs.values() if r["task"] in ("breast", "prostate_ms") and r["method"] in in_comp]
    tasks = [t for t in ["breast", "prostate_ms"] if any(r["task"] == t for r in runs)]
    methods = [m for m in make_tables.NAMES if any(r["method"] == m for r in runs)]
    _, _, summ = make_tables.summary_table(runs, tasks, methods, validation_only=False)
    rows = [[s["task"], SHORT[s["method"]], len(s["seeds"])] + [pm(*s[f"test_{k}"], nd=3) for k in make_tables.METRICS]
            for s in summ]
    subtitle("A. Test set, mean +/- s.d. over seeds (per-run value = mean over test cases)")
    table(["task", "method", "seeds", "DSC", "HD95n", "EC", "PG"], rows, left=2,
          note="HD95n: lower is better. EC = IoU@0.5 (optimised by XFedGA). PG = pointing game (Grad-CAM).")

    _, _, comp = make_tables.comparison_table(runs, tasks, pairs, args.n_boot)
    keep = []
    for c in comp:
        endpoint = make_tables._cmp_row(c["task"], c["method"], c["comparator"], c)[3]
        role = ("PRIMARY" if endpoint == "PRIMARY" else "declared 2nd" if endpoint.startswith("DECLARED") else
                "H-XAI-" + endpoint[-1] if "H-XAI" in endpoint else "exploratory" if "exploratory" in endpoint
                else "secondary")
        show = (args.all_metrics or c["metric"] in ("dsc", "hd95_norm") or role.startswith("H-XAI"))
        if show:
            keep.append([c["task"], f"{SHORT[c['method']]} vs {SHORT[c['comparator']]}", METRIC[c["metric"]], role,
                         c["n_cases"], sgn(c["mean_diff"]), ci(c["ci_low"], c["ci_high"]), c["outcome"]])
    subtitle("B. Paired differences (method - comparator), seeds averaged within case, 95% cluster bootstrap over cases")
    table(["task", "comparison", "metric", "role", "cases", "mean d", "95% CI", "outcome"], keep, left=4,
          note="WIN/TIE/LOSE from the first method's side. Breast cases = patients. "
               + ("" if args.all_metrics else "Explanation rows: only the pre-registered H-XAI-1/2 (use --all-metrics)."))
    return len(runs)


def section3(_args):
    title(3, "Independent hospital (prostate_ms_runmc, server RUNMC) + breast replication (new seeds 6-10)")
    g = main_into_tempdir(g_site_runmc, "g_runmc.json")
    rows = [[SHORT[m]] + [pm(*g["per_method"][m][k]) for k in g_site_runmc.METRICS] for m in g_site_runmc.METHODS]
    subtitle("A. RUNMC test set (14 cases), seeds 1-5: mean +/- s.d. over seeds (per-run value = mean over test images)")
    table(["method", "DSC", "HD95n", "EC", "PG"], rows)
    rows = []
    for m, role in g_site_runmc.COMPS:
        for k in g_site_runmc.METRICS:
            c = g["comparisons"][m][k]
            rows.append([f"{SHORT[m]} vs FedGA", METRIC[k], role if k == "dsc" else "", c["n_cases"],
                         sgn(c["mean_diff"]), ci(c["ci_low"], c["ci_high"]), c["outcome"], c["t_outcome"]])
    table(["comparison", "metric", "role", "cases", "mean d", "95% CI", "outcome", "t-interval"], rows, left=3,
          note=f"PRIMARY of g (DSC CI > 0): {'SUPPORTED' if g['primary_supported'] else 'NOT supported'}.")
    n_g = len(g_site_runmc.METHODS) * len(g_site_runmc.SEEDS)    # rows() asserts exactly one run per method x seed

    g2 = main_into_tempdir(g2_replication, "g2_breast.json")
    rows = []
    for k in ["dsc", "hd95_norm", "ec", "pg"]:
        c, f, x = g2["comparisons"][k], g2["per_method"]["fedga"][k], g2["per_method"]["xfedga_plus_dsc"][k]
        rows.append([METRIC[k], "PRIMARY of g2" if k == "dsc" else "secondary", pm(*f), pm(*x), c["n_cases"],
                     sgn(c["mean_diff"]), ci(c["ci_low"], c["ci_high"]), c["outcome"]])
    subtitle("B. Breast, seeds 6-10: XFedGA+ (DSC-only) vs FedGA (mean +/- s.d. over seeds; d at patient level)")
    table(["metric", "role", "FedGA", "XFedGA+ (DSC-only)", "patients", "mean d", "95% CI", "outcome"], rows, left=2,
          note=f"PRIMARY of g2 (DSC CI > 0): {'SUPPORTED' if g2['primary_supported'] else 'NOT supported'}. "
               "Same 638 test patients as the headline; only the training seeds are new.")
    return n_g + 2 * len(g2_replication.SEEDS)


def section4(args):
    title(4, "Corrupted-label dose-response (BUS-UCLM labels corrupted at p) + dishonest client (reports q_i = 1)")
    used = set()
    dr = main_into_tempdir(dose_response, "dose_response.json")
    used |= set(dr["runs"].values())
    ps, seeds = dose_response.PS, dose_response.SEEDS
    rows = []
    for m in dose_response.METHODS:
        cells = []
        for p in ps:
            v = [float(np.mean(list(dose_response.patient_dsc(dose_response.run_dir(m, s, p)).values()))) for s in seeds]
            cells.append(pm(*make_tables.mean_sd_over_seeds(v)))
        rows.append([SHORT[m]] + cells + [sgn(dr["mean_slope"][m])])
    subtitle("A. Breast test DSC by corruption rate p, seeds 1-3 (mean +/- s.d. over seeds; p = 0 reuses the headline)")
    table(["method"] + [f"p = {p}" for p in ps] + ["mean slope"], rows)
    rows = [[f"{SHORT[a]} - FedGA", v["role"], v["n_patients"], sgn(v["mean_diff_slope"]), ci(v["ci_low"], v["ci_high"]),
             outcome(v["ci_low"], v["ci_high"], +1), "yes" if v["supported"] else "no"] for a, v in dr["D1"].items()]
    subtitle("B. D1: per-patient slope of test DSC on p, paired difference vs FedGA")
    table(["slope difference", "test", "patients", "mean d", "95% CI", "outcome", "supported"], rows, left=2,
          note="WIN = DSC falls less steeply than FedGA as corruption grows. Supported if CI > 0.")
    rows = [[SHORT[m]] + [f"{v['mean_alpha_by_p'][str(p)]:.4f}" for p in ps] + [sgn(v["slope"]), ci(v["ci_low"], v["ci_high"]),
            "yes" if v["supported"] else "no"] for m, v in dr["D2"].items()]
    subtitle("C. D2 (mechanism): aggregation weight alpha of the corrupted client, mean over rounds and seeds")
    table(["method"] + [f"p = {p}" for p in ps] + ["OLS slope", "95% t-interval", "supported"], rows,
          note="Supported if the slope's interval lies below 0 (the corrupted client is down-weighted).")

    with track_post_runs(used):
        d = post.item_d()
        rows = []
        for x in d:
            clean = arm_dsc("breast", x["method"], [1, 2, 3])
            noisy = arm_dsc("breast", x["method"], [1, 2, 3], "_hrob2")
            rows.append([SHORT[x["method"]], x["role"], pm(*clean), pm(*noisy), x["n"], sgn(x["drop"]),
                         ci(x["lo"], x["hi"]), "yes" if x["supported"] else "no"])
    subtitle("D. Dishonest client (H-ROB2): BUS-UCLM always reports q_i = 1; breast, seeds 1-3")
    table(["method", "role", "clean DSC", "dishonest DSC", "patients", "drop", "95% CI", "in +/-0.02"],
          rows, left=2, note="Equivalence test: supported if the whole CI of the drop lies inside +/-0.02. "
                             "XFedGA+ scores clients itself (s_i), so it ignores the reported q_i.")
    return len(used)


def section5(args):
    title(5, "Ablations, fitness-weight screen, beta screen")
    used = set()
    with track_post_runs(used):
        e = post.item_e()
        arms = []
        for t in post.TASKS:
            for m in ["fedavg", "fedga", "xfedga", "abl_B", "abl_C", "abl_D", "abl_F", "random_search"]:
                tag = "_abl" if m.startswith("abl_") or m == "random_search" else None
                arms.append([t, SHORT[m], 3, pm(*arm_dsc(t, m, [1, 2, 3], tag))])
    subtitle("A. Ablations, test DSC, seeds 1-3 (mean +/- s.d. over seeds)")
    table(["task", "arm", "seeds", "test DSC"], arms, left=2)
    table(CMP_HEADER, cmp_rows(e), left=2, note="Paired at case level (breast: patients), 95% cluster bootstrap.")

    fw, _ = sensitivity_screens.fw_screen()
    for c in fw["cells"].values():
        used.update(c["runs"])
    cur = fw["cells"][str((0.5, 0.3, 0.2))]["mean"]
    rows = [[w + (" (current)" if w == str((0.5, 0.3, 0.2)) else ""), len(c["scores"]),
             pm(c["mean"], float(np.sqrt(c["var"]))), ", ".join(f"{s:.4f}" for s in c["scores"]), sgn(c["mean"] - cur)]
            for w, c in fw["cells"].items()]
    subtitle("B. Fitness-weight screen (w1, w2, w3), prostate_ms, XFedGA, VALIDATION only")
    table(["(w1, w2, w3)", "seeds", "val score", "per seed", "- current"], rows,
          note=f"Pooled s.d. = {fw['pooled_sd']:.4f}. Decision: {fw['outcome']}.")

    pools = [p for p in step0.load_pools(step0.HEADLINE) if p["task"] == "breast"]
    used.update(p["run_id"] for p in pools)
    rows = []
    for w in step0.NAMED:
        ch = [step0.winner(p["cands"], w) != step0.winner(p["cands"], step0.DEFAULT) for p in pools]
        win = [p["cands"][step0.winner(p["cands"], w)] for p in pools]
        rows.append([str(w), len(pools), f"{int(np.sum(ch))} ({np.mean(ch):.0%})"]
                    + [f"{np.mean([c[k] for c in win]):.4f}" for k in ("dsc", "hd95_norm", "ec")])
    subtitle("C. Fitness-weight Step 0, breast: offline re-ranking of the logged GA candidate pools (5 XFedGA runs)")
    table(["(w1, w2, w3)", "rounds", "winner changes", "winner DSC", "winner HD95n", "winner EC"], rows,
          note="Within-round selection only; winner metrics are on the fitness subset, not validation.")

    bs, _ = sensitivity_screens.beta_screen()
    used.update(r["run_id"] for r in bs["runs"])
    bruns = [r for r in beta_decision.load_runs() if r["task"] == "breast"]
    used.update(r["run_id"] for r in bruns)
    for task, dec in [("prostate_ms", bs["decision"]), ("breast", beta_decision.decide(bruns, "breast"))]:
        rows = [[f"{t['beta']}" + (" (selected)" if t["beta"] == dec["best_beta"] else ""), t["n_seeds"],
                 pm(t["score_mean"], t["score_sd"]), sgn(t["margin_over_default"]),
                 "yes" if t["beats_default_by_more_than_pooled_sd"] else "no", f"{t['q_var_mean']:.2e}"]
                for t in dec["table"]]
        subtitle(f"{'D' if task == 'prostate_ms' else 'E'}. Beta screen, {task}, XFedGA, VALIDATION only")
        table(["beta", "seeds", "val score", "- beta 0.5", "beats 0.5 by > pooled s.d.", "q_i variance"], rows,
              note=f"Pooled s.d. = {dec['pooled_sd']:.4f}. Decision: {dec['outcome']}.")
    subtitle("Val score = max centred 3-round mean validation DSC per run, mean +/- s.d. over seeds (2 seeds: exploratory).")
    return len(used)


def section6(args):
    title(6, "FedGA with the paper's GA settings, FedProx, centralised and local-only bounds")
    used = set()
    with track_post_runs(used):
        c = post.item_c()
        f, bounds = post.item_f()
        arms = []
        for t in post.TASKS:
            for m, seeds, tag in [("fedga", [1, 2, 3, 4, 5], None), ("fedga_paper", [1, 2, 3], "_paperga"),
                                  ("fedprox", [1, 2, 3, 4, 5], "_fedprox"), ("xfedga", [1, 2, 3, 4, 5], None),
                                  ("xfedga_plus", [1, 2, 3, 4, 5], None)]:
                arms.append([t, SHORT[m], len(seeds), pm(*arm_dsc(t, "fedga" if m == "fedga_paper" else m, seeds, tag))])
    subtitle("A. Test DSC per arm (mean +/- s.d. over seeds)")
    table(["task", "arm", "seeds", "test DSC"], arms, left=2)
    subtitle("B. FedGA with the paper's GA settings (seeds 1-3)")
    table(CMP_HEADER, cmp_rows(c), left=2)
    subtitle("C. FedProx (seeds 1-5)")
    table(CMP_HEADER, cmp_rows(f), left=2)
    subtitle("D. Bounds (descriptive), seeds 1-3")
    table(["task", "bound", "test DSC"], [[b["task"], b["bound"], pm(*b["dsc"])] for b in bounds], left=2,
          note="Per-run value = mean over test images (as in results/post_items/post_items.md); local-only = one model "
               "per client (test_per_image_client<k>.csv).")
    return len(used)


def section7(args):
    title(7, "Flower (multi-process federation on a single machine over loopback gRPC) vs single-process loop")
    used = set()
    with track_post_runs(used):
        b = post.item_b()
        rows = []
        for x in b:
            m = x["method"]
            fl = arm_dsc("breast", m, [1, 2, 3], "_fl")
            cl = arm_dsc("breast", m, [1, 2, 3])
            rows.append([SHORT[m], pm(*fl), pm(*cl), x["n"], sgn(x["diff"]), ci(x["ci90_lo"], x["ci90_hi"]),
                         "yes" if x["equivalent"] else "no", f"{x['max_abs_val_dsc_diff']:.1e}",
                         "ok" if x["audit_ok"] else "FAIL"])
    subtitle("Breast, seeds 1-3, test DSC mean +/- s.d. over seeds")
    table(["method", "Flower", "single-process", "patients", "d (fl-sp)", "90% CI", "equivalent", "max|d val|",
           "audit"], rows,
          note="TOST: equivalent if the 90% CI lies inside +/-0.02. max|d val| = largest per-round validation-DSC "
               "difference. Audit ok = messages carry only weights, n_i, q_i and the client index.")
    return len(used)


SECTIONS = {"1": section1, "2": section2, "3": section3, "4": section4, "5": section5, "6": section6, "7": section7}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("section", choices=list(SECTIONS) + ["all"])
    ap.add_argument("--all-metrics", action="store_true", help="section 2: show every explanation metric")
    ap.add_argument("--n_boot", type=int, default=10000, help="bootstrap resamples (10000 = as in the report)")
    args = ap.parse_args()
    timings = []
    for key in (list(SECTIONS) if args.section == "all" else [args.section]):
        t0 = time.perf_counter()
        n = SECTIONS[key](args)
        dt = time.perf_counter() - t0
        timings.append((key, n, dt))
        print(f"  Runs used: {n} run folder(s) in results/runs/   |   time: {dt:.1f} s")
        print()
    if len(timings) > 1:
        table(["section", "runs", "time (s)"], [[k, n, f"{dt:.1f}"] for k, n, dt in timings])


if __name__ == "__main__":
    main()
