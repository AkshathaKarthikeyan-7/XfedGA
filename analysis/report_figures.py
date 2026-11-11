"""
Report figures from saved data only (no training, no result file changed). Writes report_figures/:
  fig_dataset_samples.png  one image + mask overlay per dataset (first case in the loader's sorted order; prostate:
                           the slice with the largest prostate area of that case, after the 0.5 mm in-plane resampling)
  fig_flower_parity.png    per-round validation DSC, XFedGA+ breast seed 1: Flower (fl_dist, tag fl) vs the
                           single-process run (*_final), both from round_log.csv
  fig_gradcam_*.png        copied from results/figures/final/
Also prints the path of message_audit.json of the latest Flower run (by modification time).
    & $py analysis/report_figures.py
"""
import csv
import glob
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.breast_dataset import INDEXERS, _load_resized  # noqa: E402
from data.prostate_dataset import _extract_slices_multisite, _index_multisite  # noqa: E402

OUT = os.path.join(REPO, "report_figures")
RAW = os.path.join(REPO, "raw data")
BREAST = [("BUSI", "BUSI"), ("BUS-UCLM", "BUS-UCLM"), ("BUS-BRA", "BUS-BRA")]
SITES = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]
MASK_RGB = (0.89, 0.29, 0.20)


def _overlay(ax, img, mask, title):
    lo, hi = np.percentile(img, (0.5, 99.5))
    ax.imshow(np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray")
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask > 0] = (*MASK_RGB, 0.35)
    ax.imshow(rgba)
    ax.contour(mask, levels=[0.5], colors=[MASK_RGB], linewidths=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def dataset_samples():
    """3 x 3 grid: the 3 breast datasets, then the 6 prostate sites."""
    fig, axes = plt.subplots(3, 3, figsize=(9, 9.6), facecolor="white")
    rows = []
    for name, folder in BREAST:
        img_path, mask_path = INDEXERS[name](os.path.join(RAW, folder))[0]
        img, mask = _load_resized(img_path, mask_path)
        rows.append((img.astype(np.float32), mask, f"{name} ({'breast US'})\n{os.path.basename(img_path)}"))
    for site in SITES:
        vol = _index_multisite(os.path.join(RAW, "MultiSiteProstate", site))[0][0]
        slices = _extract_slices_multisite(vol)
        k = int(np.argmax([m.sum() for _, m in slices]))
        img, mask = slices[k]
        rows.append((img.astype(np.float32), mask, f"{site} (prostate MRI)\n{os.path.basename(vol)}, slice {k}"))
    for ax, (img, mask, title) in zip(axes.flat, rows):
        _overlay(ax, img, mask, title)
        print("sample:", title.replace("\n", " | "))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dataset_samples.png"), dpi=300, facecolor="white")
    plt.close(fig)


def _val_dsc(run):
    with open(os.path.join(REPO, "results", "runs", run, "round_log.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [int(r["round"]) for r in rows], [float(r["val_dsc"]) for r in rows]


def flower_parity(seed=1):
    """Two lines: single-process vs Flower per-round validation DSC."""
    fl = glob.glob(os.path.join(REPO, "results", "runs", f"breast_xfedga_plus_seed{seed}_*_fl"))
    assert len(fl) == 1, fl
    fl_run = os.path.basename(fl[0])
    custom_run = fl_run[:-len("_fl")] + "_final"
    rc, vc = _val_dsc(custom_run)
    rf, vf = _val_dsc(fl_run)
    assert rc == rf, (rc, rf)
    max_diff = max(abs(a - b) for a, b in zip(vc, vf))
    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor="white")
    ax.plot(rc, vc, color="#2a6fdb", lw=2, marker="o", ms=8, label="single-process loop")
    ax.plot(rf, vf, color="#e8590c", lw=2, ls="--", marker="o", ms=4, label="Flower (multi-process, loopback gRPC)")
    ax.set_xlabel("round")
    ax.set_ylabel("validation DSC")
    ax.set_xticks(rc)
    ax.set_title(f"XFedGA+, breast, seed {seed}: max |difference| = {max_diff:.1e}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_flower_parity.png"), dpi=300, facecolor="white")
    plt.close(fig)
    print(f"parity: {custom_run} vs {fl_run}, {len(rc)} rounds, max |diff val DSC| = {max_diff}")


def main():
    os.makedirs(OUT, exist_ok=True)
    dataset_samples()
    for f in ("fig_gradcam_prostate_ms.png", "fig_gradcam_breast.png"):
        shutil.copy2(os.path.join(REPO, "results", "figures", "final", f), os.path.join(OUT, f))
    flower_parity()
    audits = glob.glob(os.path.join(REPO, "results", "runs", "*_fl", "message_audit.json"))
    print("latest Flower message_audit.json:", os.path.abspath(max(audits, key=os.path.getmtime)))


if __name__ == "__main__":
    main()
