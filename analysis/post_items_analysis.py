"""
Pre-registered analyses of the post-headline items b, c, d, e, f (plan.md pre-registrations 3f45eab, fcf9c5c, 354b0a7,
f31ffd0, fcf54ee). Test evaluated once per run; patient (breast) / case (prostate_ms) level paired differences, seeds
averaged within patient / case, 95% cluster bootstrap (10,000) and WIN / TIE / LOSE (analysis/stats.py).
Comparator runs (as pre-registered): breast FedAvg / FedGA / XFedGA = the untagged v1 headline; breast XFedGA+ and every
prostate_ms primary arm = the *_final runs; seeds restricted to the item's seeds.
  b  Flower parity: fl_dist (tag fl) vs the same-seed custom-loop run, XFedGA and XFedGA+, seeds 1-3. Equivalent if the
     90% cluster-bootstrap CI of the per-patient test-DSC difference lies inside [-0.02, +0.02] (TOST). Also: message
     audit keys, and max |difference| of the per-round validation DSC.
  c  FedGA (paper GA settings, tag paperga) seeds 1-3: XFedGA vs it, XFedGA+ vs it, it vs FedGA (our settings).
  d  H-ROB2 (tag hrob2): drop = clean - noisy test DSC per patient (seeds 1-3); supported if the 95% CI of the drop lies
     inside [-0.02, +0.02]; XFedGA (primary of the item), XFedGA+ (secondary).
  e  Ablations (tag abl), seeds 1-3: B vs FedAvg, C vs FedGA, D vs FedGA, C vs random search, XFedGA vs F, XFedGA vs C.
  f  FedProx (seeds 1-5): XFedGA vs FedProx, XFedGA+ vs FedProx, FedProx vs FedAvg; bounds (centralized, local-only;
     seeds 1-3) descriptive: mean +- s.d. over seeds of the per-run mean test DSC (local-only: per client model).
Writes results/post_items/post_items.{json,md}.
    & $py analysis/post_items_analysis.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats import bootstrap_ci, case_means, compare, mean_sd_over_seeds, paired_differences, seed_averaged  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "post_items")
TASKS = ["breast", "prostate_ms"]
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+",
         "abl_B": "B: q_i weighting only", "abl_C": "C: FedAvg + GA (multi)", "abl_D": "D: FedGA + annealing",
         "abl_F": "F: XFedGA, random q_i", "random_search": "random search", "fedprox": "FedProx",
         "fedga_paper": "FedGA (paper GA settings)"}


def comparator_tag(task, method):
    if task == "breast" and method in ("fedavg", "fedga", "xfedga"):
        return ""
    return "_final"


def run_dir(task, method, seed, tag):
    pat = re.compile(rf"{task}_{method}_seed{seed}_[0-9a-f]{{8}}{re.escape(tag)}")
    ds = [d for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed{seed}_*")) if pat.fullmatch(os.path.basename(d))]
    assert len(ds) == 1, (task, method, seed, tag, ds)
    meta = json.load(open(os.path.join(ds[0], "run_meta.json"), encoding="utf-8"))
    assert meta.get("test_evaluated"), ds[0]
    return ds[0]


def test_rows(d, suffix=""):
    return list(csv.DictReader(open(os.path.join(d, f"test_per_image{suffix}.csv"), encoding="utf-8")))


def runs(task, method, seeds, tag=None):
    tag = comparator_tag(task, method) if tag is None else tag
    return [test_rows(run_dir(task, method, s, tag)) for s in seeds]


def cmp_row(item, task, a_lab, b_lab, c, role="secondary"):
    return {"item": item, "task": task, "comparison": f"{a_lab} vs {b_lab}", "role": role, "n": c["n_cases"],
            "diff": c["mean_diff"], "lo": c["ci_low"], "hi": c["ci_high"], "outcome": c["outcome"],
            "t_outcome": c["t_outcome"]}


def item_b():
    out, seeds = [], [1, 2, 3]
    for m in ["xfedga", "xfedga_plus"]:
        fl = [run_dir("breast", m, s, "_fl") for s in seeds]
        cl = [run_dir("breast", m, s, comparator_tag("breast", m)) for s in seeds]
        a = seed_averaged([case_means(test_rows(d), "dsc") for d in fl])
        b = seed_averaged([case_means(test_rows(d), "dsc") for d in cl])
        _, diff = paired_differences(a, b)
        lo, hi = bootstrap_ci(diff, n_boot=10000, alpha=0.10)
        maxv = 0.0
        for f, c in zip(fl, cl):
            vf = [float(r["val_dsc"]) for r in csv.DictReader(open(os.path.join(f, "round_log.csv"), encoding="utf-8"))]
            vc = [float(r["val_dsc"]) for r in csv.DictReader(open(os.path.join(c, "round_log.csv"), encoding="utf-8"))]
            maxv = max(maxv, max(abs(x - y) for x, y in zip(vf, vc)))
        aud = [json.load(open(os.path.join(f, "message_audit.json"), encoding="utf-8")) for f in fl]
        ok = all(x["metric_keys_seen"] == ["client_index", "q_i"] and x["all_shapes_ok"] for x in aud)
        out.append({"method": m, "n": len(diff), "diff": float(np.mean(diff)), "ci90_lo": lo, "ci90_hi": hi,
                    "equivalent": bool(-0.02 < lo and hi < 0.02), "max_abs_val_dsc_diff": maxv, "audit_ok": ok})
    return out


def item_c():
    out, seeds = [], [1, 2, 3]
    for t in TASKS:
        pg = runs(t, "fedga", seeds, "_paperga")
        for a, b, role in [("xfedga", "fedga_paper", "secondary"), ("xfedga_plus", "fedga_paper", "secondary"),
                           ("fedga_paper", "fedga", "secondary")]:
            ra = pg if a == "fedga_paper" else runs(t, a, seeds)
            rb = pg if b == "fedga_paper" else runs(t, b, seeds)
            out.append(cmp_row("c", t, LABEL[a], LABEL[b], compare(ra, rb, "dsc", n_boot=10000), role))
    return out


def item_d():
    out, seeds = [], [1, 2, 3]
    for m, role in [("xfedga", "primary of d"), ("xfedga_plus", "secondary")]:
        c = compare(runs("breast", m, seeds), runs("breast", m, seeds, "_hrob2"), "dsc", n_boot=10000)  # clean - noisy
        out.append({"method": m, "role": role, "n": c["n_cases"], "drop": c["mean_diff"], "lo": c["ci_low"],
                    "hi": c["ci_high"], "supported": bool(-0.02 < c["ci_low"] and c["ci_high"] < 0.02)})
    return out


def item_e():
    out, seeds = [], [1, 2, 3]
    pairs = [("abl_B", "fedavg"), ("abl_C", "fedga"), ("abl_D", "fedga"), ("abl_C", "random_search"),
             ("xfedga", "abl_F"), ("xfedga", "abl_C")]
    for t in TASKS:
        cache = {}

        def R(m):
            if m not in cache:
                cache[m] = runs(t, m, seeds, "_abl" if m.startswith("abl_") or m == "random_search" else None)
            return cache[m]
        for a, b in pairs:
            out.append(cmp_row("e", t, LABEL[a], LABEL[b], compare(R(a), R(b), "dsc", n_boot=10000)))
    return out


def item_f():
    out, bounds, seeds = [], [], [1, 2, 3, 4, 5]
    for t in TASKS:
        fp = runs(t, "fedprox", seeds, "_fedprox")
        for a, b, ra, rb in [("xfedga", "fedprox", runs(t, "xfedga", seeds), fp),
                             ("xfedga_plus", "fedprox", runs(t, "xfedga_plus", seeds), fp),
                             ("fedprox", "fedavg", fp, runs(t, "fedavg", seeds))]:
            out.append(cmp_row("f", t, LABEL[a], LABEL[b], compare(ra, rb, "dsc", n_boot=10000)))
        for m in ["centralized", "local_only"]:
            ds = [run_dir(t, m, s, "_bounds") for s in (1, 2, 3)]
            if m == "centralized":
                v = [float(np.nanmean([float(x["dsc"]) for x in test_rows(d)])) for d in ds]
                bounds.append({"task": t, "bound": "centralized (pooled clients)", "dsc": mean_sd_over_seeds(v)})
            else:
                k = len(glob.glob(os.path.join(ds[0], "test_per_image_client*.csv")))
                for c in range(1, k + 1):
                    v = [float(np.nanmean([float(x["dsc"]) for x in test_rows(d, f"_client{c}")])) for d in ds]
                    name = json.load(open(os.path.join(ds[0], "run_meta.json"), encoding="utf-8"))["clients"][c - 1]
                    bounds.append({"task": t, "bound": f"local-only, client {name}", "dsc": mean_sd_over_seeds(v)})
    return out, bounds


def main():
    b = item_b()
    c = item_c()
    d = item_d()
    e = item_e()
    f, bounds = item_f()
    res = {"b": b, "c": c, "d": d, "e": e, "f": f, "bounds": bounds}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "post_items.json"), "w", encoding="utf-8"), indent=1)
    L = ["### b. Flower parity (breast; fl_dist vs custom loop, seeds 1–3; TOST ±0.02)", "",
         "| Method | Patients | Δ test DSC (fl − custom) | 90% CI | Equivalent | Max \\|Δ val DSC per round\\| | Message audit |",
         "|---|---|---|---|---|---|---|"]
    for x in b:
        L.append(f"| {LABEL[x['method']]} | {x['n']} | {x['diff']:+.4f} | [{x['ci90_lo']:+.4f}, {x['ci90_hi']:+.4f}] | "
                 f"{'yes' if x['equivalent'] else 'no'} | {x['max_abs_val_dsc_diff']:.2e} | {'ok' if x['audit_ok'] else 'FAIL'} |")

    def table(title, rows):
        T = [title, "", "| Task | Comparison | Role | Cases | Δ test DSC | 95% CI | Outcome | t-interval |",
             "|---|---|---|---|---|---|---|---|"]
        T += [f"| {r['task']} | {r['comparison']} | {r['role']} | {r['n']} | {r['diff']:+.4f} | [{r['lo']:+.4f}, "
              f"{r['hi']:+.4f}] | **{r['outcome']}** | {r['t_outcome']} |" for r in rows]
        return T
    L += [""] + table("### c. FedGA with the paper's GA settings (seeds 1–3)", c)
    L += ["", "### d. H-ROB2: noisy q_i (BUS-UCLM reports q_i = 1), breast, seeds 1–3", "",
          "| Method | Role | Patients | Drop (clean − noisy) | 95% CI | Supported (inside ±0.02) |", "|---|---|---|---|---|---|"]
    L += [f"| {LABEL[x['method']]} | {x['role']} | {x['n']} | {x['drop']:+.4f} | [{x['lo']:+.4f}, {x['hi']:+.4f}] | "
          f"{'yes' if x['supported'] else 'no'} |" for x in d]
    L += [""] + table("### e. Ablations (seeds 1–3)", e)
    L += [""] + table("### f. FedProx (seeds 1–5)", f)
    L += ["", "**Bounds (descriptive; seeds 1–3; test DSC mean ± s.d. over seeds):**", "", "| Task | Bound | Test DSC |", "|---|---|---|"]
    L += [f"| {x['task']} | {x['bound']} | {x['dsc'][0]:.4f} ± {x['dsc'][1]:.4f} |" for x in bounds]
    open(os.path.join(OUT, "post_items.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
