"""
Review 2 breast results vs the corrected breast results (18 Oct 2026). CPU only; no runs, no test re-evaluation.

Review 2: the breast results table of the Review-2 deck (slide 10 of
  D:/Amrita/Project/Literature review/Panel review -2/XFedGA_Panel_Review_2.pptx), read from the file. Those numbers
  are SUPERSEDED (results/legacy/README.md; reports/audit/reaudit.md).
Corrected: the breast v1 headline runs (FedGA (baseline) and XFedGA, seeds 1-5, validation-selected checkpoint, test
  evaluated once; frozen code b556334; commit 49bf2ad), per-image test metrics in results/runs/<run>/test_per_image.csv.
  Per metric: mean +- s.d. over seeds of the per-run mean over test images; patient-level paired difference
  (XFedGA - FedGA; seeds averaged within patient) with 95% cluster-bootstrap CI (10,000) and WIN / TIE / LOSE
  (analysis/stats.py, as the headline tables).
  EC (IoU@0.5) = `ec` (the Review-2 definition); pointing game = `pg` (Grad-CAM, full-res layer, predicted target).
  EC_mass: from the declared evaluation of the same fixed checkpoints (analysis/ec_mass_headline.py ->
  results/xai_ec/<run_id>/ec_per_image.csv, every test image; 19 Oct 2026), merged by image id.
Writes reports/review/review2_vs_corrected.md and results/review/review2_vs_corrected.json.
    & $py analysis/review2_vs_corrected.py
"""
import csv
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import load_runs  # noqa: E402
from stats import compare, mean_sd_over_seeds  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECK = r"D:\Amrita\Project\Literature review\Panel review -2\XFedGA_Panel_Review_2.pptx"
SLIDE = "ppt/slides/slide10.xml"
METRICS = [("dsc", "DSC"), ("hd95_norm", "HD95 (normalised)"), ("ec", "EC (IoU@0.5)"), ("ec_mass", "EC_mass"),
           ("pg", "Pointing game")]
AUDIT = [
    (1, "Checkpoint chosen as the best round on the OOD **test** set (`results/audit/verify_checkpoints.json`)",
     "**All** accuracy and explanation numbers: breast table (slide 10: DSC, HD95, EC, pointing game), prostate table "
     "(slide 11), cross-task summary (slide 12), conclusions (slide 18). All optimistic."),
    (2, "Single unseeded runs; per-image tests on correlated images",
     "Every p-value and every Tied / Better / Worse interpretation on slides 10 and 11 (n = 937 images, n = 233 slices); "
     "slide 7 test choices. Certainty overstated; no seed variance."),
    (3, "BUS-UCLM malignant (red) masks binarised to empty: all malignant lesions dropped (audit D-1)",
     "Breast: every number on slide 10 (the BUS-UCLM client trained on benign lesions only); slide 9 BUS-UCLM empty-mask "
     "share and client size."),
    (4, "BUS-BRA fitness / OOD split per image: patients in both halves (audit D-2)",
     "Breast: every number on slide 10; slide 9 split (938 fitness / 937 OOD) and slide 8 OOD count."),
    (5, "Prostate XFedGA β unknown (deck 0.2, script default 0.5)",
     "Prostate XFedGA column of slide 11; slide 6 (β = 0.2 prostate); the prostate parts of slides 12 and 18."),
    (6, "Grad-CAM layer failed a known-answer test (old half-resolution layer, all-pixel target)",
     "Explanation Consistency and Pointing-Game rows of slides 10 and 11; slide 12 explanation conclusions; and, via q_i "
     "and the EC fitness term, every XFedGA number on slides 10 and 11."),
    (7, "PROMISE12 client cases are Decathlon server scans: 12 duplicated cases, 8 in the server sets "
        "(`reports/audit/leakage_note.md`)",
     "Prostate: every number on slide 11; prostate rows of slides 8–9; the prostate parts of slides 12 and 18. "
     "Breast numbers are not affected (`reports/audit/breast_duplicates.md`)."),
    (8, "Unequal fitness data: XFedGA scored fitness on the first 40 images of the unshuffled fitness loader, FedGA "
        "(baseline) on the whole fitness set (audit #37, D-5)",
     "Every FedGA-vs-XFedGA comparison: all rows of slides 10 and 11 and the conclusions on slides 12 and 18 (the two "
     "methods differed in fitness data as well as in method)."),
    (9, "\"Gradient-aware perturbation\" claimed but not implemented (audit §2.9)",
     "None: the phrase is not in the Review-2 deck, so no Review-2 number depends on it. It affects the novelty "
     "statement only."),
]
DECK_ROWS = {"DSC": "dsc", "HD95 (normalized)": "hd95_norm", "Explanation Consistency": "ec", "Pointing-Game": "pg"}


def deck_table():
    x = zipfile.ZipFile(DECK).read(SLIDE).decode("utf-8")
    tbl = re.search(r"<a:tbl>(.*?)</a:tbl>", x, re.S).group(1)
    rows = []
    for tr in re.findall(r"<a:tr\b.*?</a:tr>", tbl, re.S):
        cells = [" ".join(re.findall(r"<a:t>([^<]*)</a:t>", tc)).strip() for tc in re.findall(r"<a:tc\b.*?</a:tc>", tr, re.S)]
        rows.append([c.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&") for c in cells])
    header, body = rows[0], rows[1:]
    out = {DECK_ROWS[r[0]]: {"fedga": r[1], "xfedga": r[2], "interpretation": r[3]} for r in body if r[0] in DECK_ROWS}
    subtitle = re.findall(r"<a:t>(n = [^<]*)</a:t>", x)
    return header, out, (subtitle[0] if subtitle else "")


def main():
    header, deck, subtitle = deck_table()
    runs = [r for r in load_runs(os.path.join(REPO, "results", "runs", "*"), "main")
            if r["task"] == "breast" and r["method"] in ("fedga", "xfedga")]
    by = {m: sorted([r for r in runs if r["method"] == m], key=lambda r: r["seed"]) for m in ("fedga", "xfedga")}
    assert [r["seed"] for r in by["fedga"]] == [1, 2, 3, 4, 5] == [r["seed"] for r in by["xfedga"]], "need seeds 1-5"
    assert all(r["meta"].get("test_evaluated") for r in runs)
    n_img = len(by["fedga"][0]["test"])
    n_pat = len({row["group_id"] for row in by["fedga"][0]["test"]})
    res = {}
    for m in by:                                   # EC_mass from the declared evaluation of the same checkpoints
        for r in by[m]:
            path = os.path.join(REPO, "results", "xai_ec", r["run_id"], "ec_per_image.csv")
            if os.path.exists(path):
                ecm = {x["image_id"]: x["ec_mass"] for x in csv.DictReader(open(path, encoding="utf-8"))}
                for row in r["test"]:
                    row["ec_mass"] = ecm[row["image_id"]]
    for key, _ in METRICS:
        if key not in by["fedga"][0]["test"][0]:
            res[key] = None
            continue
        per = {m: [float(np.nanmean([float(row[key]) for row in r["test"]])) for r in by[m]] for m in by}
        c = compare([r["test"] for r in by["xfedga"]], [r["test"] for r in by["fedga"]], key, n_boot=10000)
        res[key] = {"fedga": mean_sd_over_seeds(per["fedga"]), "xfedga": mean_sd_over_seeds(per["xfedga"]),
                    "per_seed": per, **c}
    out = {"deck": DECK, "slide": SLIDE, "deck_header": header, "deck_subtitle": subtitle, "deck_rows": deck,
           "corrected": res, "runs": {m: [r["run_id"] for r in by[m]] for m in by},
           "n_test_images": n_img, "n_test_patients": n_pat}
    os.makedirs(os.path.join(REPO, "results", "review"), exist_ok=True)
    with open(os.path.join(REPO, "results", "review", "review2_vs_corrected.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    pct = lambda v: f"{100 * v:.1f}%"
    fmt = lambda key, v: pct(v) if key == "pg" else f"{v:.3f}"
    L = ["# Breast results: Review 2 vs corrected", "",
         "*Every number below is read from a file: the Review-2 deck (slide 10) or `results/` "
         "(`analysis/review2_vs_corrected.py` → `results/review/review2_vs_corrected.json`). No new runs, no test "
         "re-evaluation.*", "",
         f"## Review 2 (as presented; SUPERSEDED) — {subtitle}", "",
         "| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for key, label in METRICS:
        if key in deck:
            d = deck[key]
            L.append(f"| {label} | {d['fedga']} | {d['xfedga']} | {d['interpretation']} |")
    L += ["", "Review 2 setup: checkpoint = best round on the OOD **test** set; one unseeded run per method; per-image "
          "tests on correlated images; BUS-UCLM malignant masks dropped; BUS-BRA fitness / test split per image "
          "(patients in both); old Grad-CAM layer.", "",
          f"## Corrected (breast v1 headline; {n_img} test images from {n_pat} BUS-BRA patients; 5 seeds)", "",
          "| Metric | FedGA (baseline) | XFedGA | XFedGA − FedGA (patient-level) | 95% CI (cluster bootstrap) | Outcome |",
          "|---|---|---|---|---|---|"]
    for key, label in METRICS:
        r = res[key]
        if r is None:
            L.append(f"| {label} | not available | not available | — | — | not computed: needs a test re-evaluation "
                     f"(EC_mass was introduced on 18 Oct 2026) |")
            continue
        sc = 100 if key == "pg" else 1
        unit = " pp" if key == "pg" else ""
        L.append(f"| {label} | {fmt(key, r['fedga'][0])} ± {fmt(key, r['fedga'][1])} | {fmt(key, r['xfedga'][0])} ± "
                 f"{fmt(key, r['xfedga'][1])} | {sc * r['mean_diff']:+.{1 if key == 'pg' else 4}f}{unit} | "
                 f"[{sc * r['ci_low']:+.{1 if key == 'pg' else 4}f}, {sc * r['ci_high']:+.{1 if key == 'pg' else 4}f}]{unit} | "
                 f"{r['outcome']} |")
    L += ["", "WIN / TIE / LOSE are from XFedGA's point of view (lower HD95 is better). Mean ± s.d. over the 5 seeds of "
          "the per-run mean over test images. The pointing game and EC here use the corrected Grad-CAM (full-resolution "
          "decoder layer, predicted-foreground target)."]
    L += ["", "## Which Review-2 numbers each audit item affects", "",
          "Items 1–9 follow `reports/audit/reaudit.md` (thesis Chapter 4 numbering, confirmed 18 Oct 2026). Slide "
          "numbers refer to the Review-2 deck.", "",
          "| Item | Issue | Review-2 numbers affected |", "|---|---|---|"] + [f"| {i} | {issue} | {aff} |" for i, issue, aff in AUDIT]
    os.makedirs(os.path.join(REPO, "reports", "review"), exist_ok=True)
    with open(os.path.join(REPO, "reports", "review", "review2_vs_corrected.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
