"""
Final figures (25 Oct 2026). Static PNGs for the thesis / slides; every number is read from results/runs (test
evaluated once per run, as pre-registered). Palette: the dataviz reference categorical palette in fixed slot order,
one hue per METHOD (never per rank), markers as a second encoding (the palette validator needs Node, not installed).
  1 fig_dsc_per_method.png   test DSC per method, per task (breast, prostate_ms, prostate_ms_runmc): per-seed run means
                             (dots) and the mean +- s.d. over seeds
  2 fig_dose_response.png    mean test DSC vs corruption share p (breast), mean +- s.d. over seeds 1-3 (band)
  3 fig_alpha_corrupted.png  alpha of the corrupted client (BUS-UCLM) per round, XFedGA and XFedGA+, one panel per p,
                             mean +- s.d. over seeds 1-3
  4 fig_gradcam_<task>.png   Grad-CAM examples. SELECTION RULE (fixed before any image was viewed): test images ranked by
                             the FedGA (baseline) seed-1 per-image test DSC; rows = best, median, worst (ties by image
                             id). Columns: image + ground truth, then FedGA, XFedGA, XFedGA+, XFedGA+ (DSC-only) seed-1
                             checkpoints: prediction contour + Grad-CAM (full-res layer, predicted target).
Writes results/figures/final/*.png.
    & $py analysis/final_figures.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
RUNS = os.path.join(REPO, "results", "runs")
OUT = os.path.join(REPO, "results", "figures", "final")
COLOR = {"fedavg": "#2a78d6", "fedga": "#eb6834", "xfedga": "#1baf7a", "xfedga_plus": "#eda100",
         "xfedga_plus_dsc": "#e87ba4"}          # reference palette slots 1-5, fixed per method
MARK = {"fedavg": "o", "fedga": "s", "xfedga": "^", "xfedga_plus": "D", "xfedga_plus_dsc": "v"}
LABEL = {"fedavg": "FedAvg", "fedga": "FedGA (baseline)", "xfedga": "XFedGA", "xfedga_plus": "XFedGA+",
         "xfedga_plus_dsc": "XFedGA+ (DSC-only)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
CAM = LinearSegmentedColormap.from_list("cam_blue", ["#ffffff00", "#86b6ef", "#2a78d6", "#104281"])
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def run_dir(task, method, seed, tag):
    pat = re.compile(rf"{task}_{method}_seed{seed}_[0-9a-f]{{8}}{re.escape(tag)}")
    ds = [d for d in glob.glob(os.path.join(RUNS, f"{task}_{method}_seed{seed}_*")) if pat.fullmatch(os.path.basename(d))]
    assert len(ds) == 1, (task, method, seed, tag, ds)
    return ds[0]


def tag_of(task, method):
    if task == "breast":
        return "" if method in ("fedavg", "fedga", "xfedga") else "_final"
    return "_runmc" if task == "prostate_ms_runmc" else "_final"


def mean_test(d, k="dsc"):
    return float(np.nanmean([float(r[k]) for r in csv.DictReader(open(os.path.join(d, "test_per_image.csv"), encoding="utf-8"))]))


def fig_dsc():
    tasks = [("breast", "breast (v1)", ["fedavg", "fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]),
             ("prostate_ms", "prostate_ms (v2; server BMC)", ["fedavg", "fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]),
             ("prostate_ms_runmc", "prostate_ms_runmc (v2; server RUNMC)", ["fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (t, title, ms) in zip(axes, tasks):
        for i, m in enumerate(ms):
            v = np.array([mean_test(run_dir(t, m, s, tag_of(t, m))) for s in range(1, 6)])
            ax.scatter(np.full(5, i) + np.linspace(-0.12, 0.12, 5), v, s=14, color=COLOR[m], alpha=0.55,
                       marker=MARK[m], edgecolors="white", linewidths=0.5, zorder=3)
            ax.errorbar(i + 0.28, v.mean(), yerr=v.std(ddof=1), fmt=MARK[m], color=COLOR[m], ms=8, lw=2, capsize=0,
                        mec="white", mew=1.2, zorder=4)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL[m] for m in ms], rotation=30, ha="right")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test DSC (per-seed run mean; mean ± s.d.)")
    fig.suptitle("Test DSC per method and task (5 seeds; test evaluated once per run)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(OUT, "fig_dsc_per_method.png"))
    plt.close(fig)


def fig_dose():
    ps, tags = [0.0, 0.25, 0.5, 1.0], {0.25: "_dose_p025", 0.5: "_dose_p050", 1.0: "_dose_p100"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ends = {}
    for m in ["fedavg", "fedga", "xfedga", "xfedga_plus"]:
        v = np.array([[mean_test(run_dir("breast", m, s, tags.get(p, tag_of("breast", m)))) for s in (1, 2, 3)] for p in ps])
        mu, sd = v.mean(1), v.std(1, ddof=1)
        ax.fill_between(ps, mu - sd, mu + sd, color=COLOR[m], alpha=0.12, lw=0)
        ax.plot(ps, mu, color=COLOR[m], lw=2, marker=MARK[m], ms=7, mec="white", mew=1.2, label=LABEL[m])
        ends[m] = mu[-1]
    ys, gap = {}, 0.0065                      # end labels, spread to a minimum vertical gap (no collisions)
    for m in sorted(ends, key=ends.get):
        y = ends[m]
        if ys and y - max(ys.values()) < gap:
            y = max(ys.values()) + gap
        ys[m] = y
    for m, y in ys.items():
        ax.annotate(LABEL[m], (1.0, ends[m]), xytext=(1.04, y), textcoords="data", va="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5))
    ax.set_xticks(ps)
    ax.set_xlabel("share p of BUS-UCLM training images with corrupted masks")
    ax.set_ylabel("test DSC (mean ± s.d. over seeds 1–3)")
    ax.set_title("Dose-response, breast (v1)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_xlim(-0.03, 1.25)
    fig.savefig(os.path.join(OUT, "fig_dose_response.png"))
    plt.close(fig)


def fig_alpha():
    ps, tags = [0.0, 0.25, 0.5, 1.0], {0.25: "_dose_p025", 0.5: "_dose_p050", 1.0: "_dose_p100"}
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, p in zip(axes, ps):
        for m in ["xfedga", "xfedga_plus"]:
            a = np.array([[json.loads(r["alpha_i"])[1] for r in csv.DictReader(open(os.path.join(
                run_dir("breast", m, s, tags.get(p, tag_of("breast", m))), "round_log.csv"), encoding="utf-8"))]
                for s in (1, 2, 3)])
            x = np.arange(1, a.shape[1] + 1)
            mu, sd = a.mean(0), a.std(0, ddof=1)
            ax.fill_between(x, mu - sd, mu + sd, color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(x, mu, color=COLOR[m], lw=2, marker=MARK[m], ms=4, markevery=3, label=LABEL[m])
        ax.set_title(f"p = {p}", fontsize=9, loc="left")
        ax.set_xlabel("round")
    axes[0].set_ylabel("alpha of BUS-UCLM\n(mean ± s.d., seeds 1–3)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Aggregation weight of the corrupted client per round (breast)", x=0.01, ha="left", fontsize=10)
    fig.savefig(os.path.join(OUT, "fig_alpha_corrupted.png"))
    plt.close(fig)


def fig_gradcam(task):
    import torch
    import yaml
    from experiments.config import load_config
    from experiments.tasks import build_task
    from models.residual_unet import build_residual_unet
    from xai.explainability import cam_layer, get_gradcam
    ms = ["fedga", "xfedga", "xfedga_plus", "xfedga_plus_dsc"]
    ref = run_dir(task, "fedga", 1, tag_of(task, "fedga"))
    rows = [r for r in csv.DictReader(open(os.path.join(ref, "test_per_image.csv"), encoding="utf-8"))
            if r["dsc"] not in ("", "nan")]
    rows.sort(key=lambda r: (-float(r["dsc"]), r["image_id"]))
    pick = {"best": rows[0], "median": rows[len(rows) // 2], "worst": rows[-1]}
    cfg = yaml.safe_load(open(os.path.join(ref, "config_resolved.yaml"), encoding="utf-8"))
    base = load_config(task)
    for k in ("preprocessing", "fitness", "image_size"):
        if k in cfg:
            base[k] = cfg[k]
    test = build_task(task, base, part="server").server["test"]
    idx = {iid: i for i, iid in enumerate(test.image_ids)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for m in ms:
        d = run_dir(task, m, 1, tag_of(task, m))
        net = build_residual_unet().to(dev)
        net.load_state_dict(torch.load(glob.glob(os.path.join(d, "*_valselected.pt"))[0], map_location=dev))
        net.eval()
        models[m] = net
    fig, axes = plt.subplots(3, 1 + len(ms), figsize=(2.1 * (1 + len(ms)), 6.6))
    rec = {}
    for r, (name, row) in enumerate(pick.items()):
        x, y = test.dataset[idx[row["image_id"]]]
        img, gt = x[0].numpy(), y.numpy()
        ax = axes[r, 0]
        ax.imshow(img, cmap="gray")
        ax.contour(gt, levels=[0.5], colors=[INK], linewidths=1.0)
        ax.set_title(f"{name}: {row['image_id']}\nground truth", fontsize=7)
        rec[name] = {"image_id": row["image_id"], "fedga_seed1_dsc": float(row["dsc"])}
        for c, m in enumerate(ms, start=1):
            net = models[m]
            with torch.no_grad():
                pred = torch.argmax(net(x[None].to(dev)), 1)[0].cpu().numpy()
            cam = get_gradcam(net, cam_layer(net, base["fitness"]["cam_layer"]), x[None].to(dev),
                              target_class=1, cam_target=base["fitness"]["cam_target"])
            inter, s = float((pred * gt).sum()), float(pred.sum() + gt.sum())
            dsc = 1.0 if s == 0 else 2 * inter / s
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            ax.imshow(cam, cmap=CAM, alpha=0.75, vmin=0, vmax=1)
            ax.contour(gt, levels=[0.5], colors=[INK], linewidths=0.8, linestyles="dashed")
            if pred.any():
                ax.contour(pred, levels=[0.5], colors=[COLOR[m]], linewidths=1.4)
            ax.set_title(f"{LABEL[m]}\nDSC {dsc:.3f}", fontsize=7)
            rec[name][m] = dsc
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Grad-CAM examples, {task} (seed-1 checkpoints; rows chosen by FedGA seed-1 test DSC: best / median / "
                 f"worst). Dashed = ground truth; solid = prediction; blue = Grad-CAM.", x=0.01, ha="left", fontsize=8)
    fig.savefig(os.path.join(OUT, f"fig_gradcam_{task}.png"))
    plt.close(fig)
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_dsc()
    fig_dose()
    fig_alpha()
    rec = {t: fig_gradcam(t) for t in ["breast", "prostate_ms"]}
    json.dump(rec, open(os.path.join(OUT, "gradcam_examples.json"), "w", encoding="utf-8"), indent=1)
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
