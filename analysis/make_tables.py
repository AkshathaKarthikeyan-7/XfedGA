"""
Results tables and figures from results/runs/* (every number traces to a run folder).

    & $py analysis/make_tables.py --group main --out results/tables/headline
    & $py analysis/make_tables.py --group main --comparisons ablations --out results/tables/ablations
    & $py analysis/make_tables.py --group hetero --out results/tables/hetero
    & $py analysis/make_tables.py --group "betascreen_*" --validation_only --out <dir>     (no test data needed)

A run's group is its tag (run id suffix after the config hash; "main" if none), so the headline
(no tag), the heterogeneous partition (hetero), robustness (noisyq, badlabels, dp) and the beta screen
(betascreen_b*) never mix. Outputs in --out:
  summary.{md,tex}       per task x method: seeds, test DSC / HD95_norm / EC / pointing game as
                         mean +/- s.d. over seeds (per-run value = mean over test cases), selected round
  comparisons.{md,tex}   case-level paired differences, 95% cluster-bootstrap CI over cases, Cohen's d_z,
                         WIN / TIE / LOSE (analysis/stats.py)
  outcome.{md,tex}       spec 9B outcome table: accuracy, explanation (independent metrics), stability, cost
  fig_val_dsc_<task>.png, fig_qi_alpha_<task>.png   validation DSC, q_i and alpha_i vs round (mean +/- s.d.)
  tables.json            everything above, machine-readable
"""
import argparse
import csv
import fnmatch
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import DIRECTION, case_means, compare, mean_sd_over_seeds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA", "abl_B": "B: weighted agg. only",
         "abl_C": "C: FedAvg + GA (multi)", "abl_D": "D: FedGA + annealing", "abl_F": "F: XFedGA, random $q_i$",
         "random_search": "random search", "fedprox": "FedProx", "centralized": "centralized", "local_only": "local only"}
COMPARISONS = {
    "headline": [("xfedga", "fedga"), ("xfedga", "fedavg"), ("fedga", "fedavg")],
    "ablations": [("abl_B", "fedavg"), ("abl_C", "fedga"), ("abl_D", "fedga"), ("abl_C", "random_search"),
                  ("xfedga", "abl_F"), ("xfedga", "abl_C")],
    "baselines": [("xfedga", "fedprox"), ("fedprox", "fedavg")],
}
METRICS = ["dsc", "hd95_norm", "ec", "pg"]
XAI_METRICS = ["pg_gradcam", "pg_gradcampp", "deletion_auc", "insertion_auc"]


def run_group(run_id):
    parts = run_id.split("_")
    i = next(k for k, p in enumerate(parts) if p.startswith("seed"))
    tail = parts[i + 2:]          # after seed<k> and the 8-char config hash
    return "_".join(tail) if tail else "main"


def load_runs(pattern, group):
    runs = []
    for meta_path in glob.glob(os.path.join(pattern, "run_meta.json")):
        d = os.path.dirname(meta_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        rid = meta.get("run_id", os.path.basename(d))
        g = run_group(rid)
        if not fnmatch.fnmatch(g, group):
            continue
        r = {"dir": d, "run_id": rid, "group": g, "task": meta["task"], "method": meta["method"], "seed": meta["seed"],
             "meta": meta, "rounds": list(csv.DictReader(open(os.path.join(d, "round_log.csv"), encoding="utf-8")))
             if os.path.exists(os.path.join(d, "round_log.csv")) else []}
        tp = os.path.join(d, "test_per_image.csv")
        r["test"] = list(csv.DictReader(open(tp, encoding="utf-8"))) if os.path.exists(tp) else None
        xp = os.path.join(REPO, "results", "xai", rid, "xai_per_image.csv")
        r["xai"] = list(csv.DictReader(open(xp, encoding="utf-8"))) if os.path.exists(xp) else None
        runs.append(r)
    return runs


def by(runs, task, method, key=None):
    return [r for r in runs if r["task"] == task and r["method"] == method and (key is None or r.get(key) is not None)]


def run_value(rows, metric):
    cm = case_means(rows, metric)
    return float(np.mean(list(cm.values()))) if cm else float("nan")


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}" if np.isfinite(s) else (f"{m:.{nd}f}" if np.isfinite(m) else "–")


def latex_escape(s):
    return str(s).replace("±", r"$\pm$").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_table(path_stem, header, rows, caption=""):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    with open(path_stem + ".md", "w", encoding="utf-8") as f:
        f.write((f"{caption}\n\n" if caption else "") + "\n".join(md) + "\n")
    tex = [r"\begin{tabular}{" + "l" * 2 + "c" * (len(header) - 2) + "}", r"\hline",
           " & ".join(latex_escape(h) for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(latex_escape(c) for c in r) + r" \\" for r in rows] + [r"\hline", r"\end{tabular}"]
    with open(path_stem + ".tex", "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")


def summary_table(runs, tasks, methods, validation_only):
    rows, data = [], []
    for task in tasks:
        for m in methods:
            rs = by(runs, task, m)
            if not rs:
                continue
            sel = mean_sd_over_seeds([r["meta"].get("selected_round", np.nan) for r in rs])
            val = mean_sd_over_seeds([max(r["meta"]["smoothed_val_dsc"]) for r in rs if r["meta"].get("smoothed_val_dsc")])
            rec = {"task": task, "method": m, "groups": sorted({r["group"] for r in rs}), "seeds": sorted(r["seed"] for r in rs),
                   "val_score": val, "selected_round": sel}
            cells = [task, NAMES.get(m, m), len(rs), fmt(*val), fmt(*sel, nd=1)]
            if not validation_only:
                for metric in METRICS:
                    vals = [run_value(r["test"], metric) for r in rs if r["test"]]
                    rec[f"test_{metric}"] = mean_sd_over_seeds(vals) if vals else (float("nan"), float("nan"))
                    cells.append(fmt(*rec[f"test_{metric}"]))
            rows.append(cells)
            data.append(rec)
    header = ["task", "method", "seeds", "val score", "sel. round"] + ([] if validation_only else
                                                                        ["test DSC", "test HD95$_{norm}$", "test EC", "test PG"])
    return header, rows, data


def comparison_table(runs, tasks, pairs, n_boot):
    rows, data = [], []
    for task in tasks:
        for a, b in pairs:
            ra, rb = by(runs, task, a, "test"), by(runs, task, b, "test")
            if not ra or not rb:
                continue
            for metric in ["dsc", "hd95_norm"]:
                c = compare([r["test"] for r in ra], [r["test"] for r in rb], metric, n_boot=n_boot)
                data.append({"task": task, "method": a, "comparator": b, **c})
                rows.append([task, f"{NAMES.get(a, a)} vs {NAMES.get(b, b)}", metric, c["n_cases"],
                             f"{c['mean_diff']:+.4f}", f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]",
                             f"{c['effect_dz']:+.2f}" if np.isfinite(c["effect_dz"]) else "–", c["outcome"]])
            xa, xb = by(runs, task, a, "xai"), by(runs, task, b, "xai")
            for metric in XAI_METRICS if xa and xb else []:
                c = compare([r["xai"] for r in xa], [r["xai"] for r in xb], metric, n_boot=n_boot)
                data.append({"task": task, "method": a, "comparator": b, **c})
                rows.append([task, f"{NAMES.get(a, a)} vs {NAMES.get(b, b)}", metric, c["n_cases"],
                             f"{c['mean_diff']:+.4f}", f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]",
                             f"{c['effect_dz']:+.2f}" if np.isfinite(c["effect_dz"]) else "–", c["outcome"]])
    header = ["task", "comparison", "metric", "cases", "mean diff.", "95% CI (cluster bootstrap)", "$d_z$", "outcome"]
    return header, rows, data


def outcome_table(runs, tasks, comp_data, method="xfedga", comparator="fedga"):
    rows = []
    for task in tasks:
        def pick(metric):
            return next((c for c in comp_data if c["task"] == task and c["method"] == method
                         and c["comparator"] == comparator and c["metric"] == metric), None)
        acc = pick("dsc")
        acc_s = f"{acc['outcome']} (ΔDSC {acc['mean_diff']:+.3f}, CI [{acc['ci_low']:+.3f}, {acc['ci_high']:+.3f}])" if acc else "–"
        ex = [c for c in (pick(m) for m in ["pg_gradcam", "deletion_auc"]) if c]
        ex_s = "; ".join(f"{c['metric']}: {c['outcome']} ({c['mean_diff']:+.3f}, CI [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}])"
                         for c in ex) if ex else "pending (independent XAI metrics not computed)"
        sd = {m: mean_sd_over_seeds([run_value(r["test"], "dsc") for r in by(runs, task, m, "test")])[1]
              for m in (method, comparator)}
        last5 = {m: float(np.mean([np.std([float(x["val_dsc"]) for x in r["rounds"][-5:]]) for r in by(runs, task, m)]))
                 if by(runs, task, m) else float("nan") for m in (method, comparator)}
        stab_s = (f"seed s.d. {sd[method]:.3f} vs {sd[comparator]:.3f}; last-5-round val s.d. "
                  f"{last5[method]:.3f} vs {last5[comparator]:.3f}")
        t = {m: float(np.mean([np.mean([float(x["round_s"]) for x in r["rounds"]]) for r in by(runs, task, m)]))
             if by(runs, task, m) else float("nan") for m in (method, comparator)}
        cost_s = f"{t[method]:.0f} s vs {t[comparator]:.0f} s per round ({(t[method] / t[comparator] - 1) * 100:+.0f}%)" \
            if np.isfinite(t[method]) and np.isfinite(t[comparator]) and t[comparator] > 0 else "–"
        rows.append([task, acc_s, ex_s, stab_s, cost_s])
    header = ["task", "accuracy", "explanation (independent metrics)", "stability", "cost"]
    return header, rows


def figures(runs, tasks, methods, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in tasks:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in methods:
            rs = [r for r in by(runs, task, m) if r["rounds"]]
            if not rs:
                continue
            n = min(len(r["rounds"]) for r in rs)
            v = np.array([[float(x["val_dsc"]) for x in r["rounds"][:n]] for r in rs])
            mu, sd = v.mean(0), v.std(0, ddof=1) if len(rs) > 1 else np.zeros(n)
            x = np.arange(1, n + 1)
            ax.plot(x, mu, label=f"{NAMES.get(m, m)} (n={len(rs)})".replace("$", ""))
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.2)
        ax.set_xlabel("round"); ax.set_ylabel("validation DSC"); ax.set_title(f"{task}: validation DSC (mean ± s.d. over seeds)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_val_dsc_{task}.png"), dpi=150)
        plt.close(fig)
        rs = [r for r in runs if r["task"] == task and r["rounds"] and json.loads(r["rounds"][0]["q_i"] or "[]")]
        if not rs:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key, ax in zip(["q_i", "alpha_i"], axes):
            for lab in sorted({(r["method"], r["group"]) for r in rs}):
                sel = [r for r in rs if (r["method"], r["group"]) == lab]
                n = min(len(r["rounds"]) for r in sel)
                arr = np.array([[json.loads(x[key]) for x in r["rounds"][:n]] for r in sel])   # seeds x rounds x clients
                for k in range(arr.shape[2]):
                    mu = arr[:, :, k].mean(0)
                    ax.plot(np.arange(1, n + 1), mu, label=f"{NAMES.get(lab[0], lab[0])} [{lab[1]}] client {k + 1}".replace("$", ""))
            ax.set_xlabel("round"); ax.set_ylabel(key); ax.set_title(f"{task}: {key} per client (mean over seeds)")
            ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig_qi_alpha_{task}.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(REPO, "results", "runs", "*"))
    ap.add_argument("--group", default="main", help="run tag group (glob), e.g. main, hetero, 'betascreen_*'")
    ap.add_argument("--comparisons", default="headline", choices=sorted(COMPARISONS))
    ap.add_argument("--validation_only", action="store_true", help="no test data: validation curves and scores only")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = load_runs(args.runs, args.group)
    if not runs:
        print("no runs match")
        return
    os.makedirs(args.out, exist_ok=True)
    tasks = [t for t in ["prostate", "breast"] if any(r["task"] == t for r in runs)]
    methods = [m for m in NAMES if any(r["method"] == m for r in runs)]
    header, rows, summary = summary_table(runs, tasks, methods, args.validation_only)
    write_table(os.path.join(args.out, "summary"), header, rows,
                "Per task and method: mean ± s.d. over seeds (per-run value = mean over test cases).")
    out = {"group": args.group, "runs": sorted(r["run_id"] for r in runs), "summary": summary}
    if not args.validation_only:
        h, r_, comp = comparison_table(runs, tasks, COMPARISONS[args.comparisons], args.n_boot)
        write_table(os.path.join(args.out, "comparisons"), h, r_,
                    "Case-level paired differences (seeds averaged within case), 95% cluster-bootstrap CI over cases.")
        h, r_ = outcome_table(runs, tasks, comp)
        write_table(os.path.join(args.out, "outcome"), h, r_, "Outcome (spec 9B): XFedGA vs FedGA (baseline).")
        out["comparisons"] = comp
    figures(runs, tasks, methods, args.out)
    with open(os.path.join(args.out, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"{len(runs)} runs ({', '.join(tasks)}); outputs in {args.out}")


if __name__ == "__main__":
    main()
