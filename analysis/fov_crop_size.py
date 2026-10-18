"""
Protocol v2 prostate FOV crop size S (pre-registered rule, plan.md, commit 661a173): after the 0.5 mm in-plane
resampling, a square of side S mm centred on the IMAGE CENTRE (no ground truth at inference). S = the smallest
multiple of 10 mm containing every gland pixel of the CLIENT TRAINING masks (PROMISE12, all 50 cases).
Reported check: share of Decathlon FITNESS and VALIDATION gland pixels that the crop would cut (test not read).
Writes results/audit/fov_crop.json.
--task prostate_ms (18 Oct 2026): S from the multi-site CLIENT masks only (BIDMC, HK, I2CVB, RUNMC, UCL), checked on
the BMC fitness and validation cases (test not read). Writes results/audit/fov_crop_prostate_ms.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import SimpleITK as sitk
import nibabel as nib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from data.prostate_dataset import (TARGET_SPACING_MM, _index_decathlon, _index_multisite,  # noqa: E402
                                   _index_promise12, _multisite_mask_path, _resample_to_spacing_nib_nearest,
                                   _resample_to_spacing_sitk)
from experiments.config import load_config  # noqa: E402
from experiments.tasks import build_task  # noqa: E402


def promise_mask(path):
    m = sitk.ReadImage(path.replace(".mhd", "_segmentation.mhd"))
    return sitk.GetArrayFromImage(_resample_to_spacing_sitk(m, is_mask=True)) > 0       # (z, y, x)


def decathlon_mask(path):
    lab = nib.load(path.replace("imagesTr", "labelsTr"))
    m = _resample_to_spacing_nib_nearest(lab.get_fdata(), lab.header.get_zooms()[:2]) > 0   # (x, y, z)
    return np.moveaxis(m, 2, 0)                                                        # (z, x, y)


def half_extent_mm(mask):
    """max distance (mm, Chebyshev) of any gland pixel from the in-plane image centre."""
    z, a, b = np.nonzero(mask)
    ca, cb = (mask.shape[1] - 1) / 2, (mask.shape[2] - 1) / 2
    return float(max(np.abs(a - ca).max(), np.abs(b - cb).max()) + 0.5) * TARGET_SPACING_MM


def cut_share(mask, side_mm):
    h = side_mm / 2 / TARGET_SPACING_MM
    ca, cb = (mask.shape[1] - 1) / 2, (mask.shape[2] - 1) / 2
    a = np.abs(np.arange(mask.shape[1]) - ca)[:, None] <= h
    b = np.abs(np.arange(mask.shape[2]) - cb)[None, :] <= h
    inside = mask & (a & b)[None]
    return 1.0 - inside.sum() / max(mask.sum(), 1)


def multisite_mask(path):
    m = sitk.ReadImage(_multisite_mask_path(path))
    return sitk.GetArrayFromImage(_resample_to_spacing_sitk(m, is_mask=True)) > 0       # (z, y, x)


def main_ms():
    cfg = load_config("prostate_ms")
    train = {f"{c['name']}/{os.path.basename(p).split('.')[0]}": half_extent_mm(multisite_mask(p))
             for c in cfg["clients"] for p, _ in _index_multisite(c["path"])}
    side = int(math.ceil(2 * max(train.values()) / 10.0) * 10)
    srv = build_task("prostate_ms", cfg, part="server").split_log["server"]
    cases = {os.path.basename(p).split(".")[0]: p for p, _ in _index_multisite(cfg["server"]["path"])}
    check = {}
    for role in ["fitness", "validation"]:          # the test set is not read
        shares = {c: cut_share(multisite_mask(cases[c]), side) for c in srv[role]["cases"]}
        check[role] = {"cases": len(shares), "max_share_gland_cut": max(shares.values()),
                       "cases_with_any_cut": [c for c, v in shares.items() if v > 0]}
    out = {"rule": "smallest multiple of 10 mm >= 2 x max Chebyshev distance of any multi-site CLIENT gland pixel "
                   "from the in-plane image centre, after 0.5 mm resampling",
           "train_half_extent_mm": {"max": max(train.values()), "median": float(np.median(list(train.values()))),
                                    "argmax": max(train, key=train.get), "per_case": train},
           "side_mm": side, "side_px_at_0.5mm": int(side / TARGET_SPACING_MM), "check_server_BMC": check}
    with open(os.path.join("results", "audit", "fov_crop_prostate_ms.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "train_half_extent_mm"}, indent=1),
          {k: v for k, v in out["train_half_extent_mm"].items() if k != "per_case"})


def main():
    train = {os.path.basename(p): half_extent_mm(promise_mask(p)) for p, _ in _index_promise12("raw data/PROMISE12")}
    side = int(math.ceil(2 * max(train.values()) / 10.0) * 10)
    cfg = load_config("prostate")
    srv = build_task("prostate", cfg, part="server").split_log["server"]
    cases = {os.path.basename(p).replace(".nii.gz", ""): p for p, _ in _index_decathlon(cfg["server"]["path"])}
    check = {}
    for role in ["fitness", "validation"]:          # the test set is not read
        shares = {c: cut_share(decathlon_mask(cases[c]), side) for c in srv[role]["cases"]}
        check[role] = {"cases": len(shares), "max_share_gland_cut": max(shares.values()),
                       "cases_with_any_cut": [c for c, v in shares.items() if v > 0]}
    out = {"rule": "smallest multiple of 10 mm >= 2 x max Chebyshev distance of any PROMISE12 (client training) gland "
                   "pixel from the in-plane image centre, after 0.5 mm resampling",
           "train_half_extent_mm": {"max": max(train.values()), "median": float(np.median(list(train.values()))),
                                    "per_case": train},
           "side_mm": side, "side_px_at_0.5mm": int(side / TARGET_SPACING_MM), "check_server": check}
    os.makedirs(os.path.join("results", "audit"), exist_ok=True)
    with open(os.path.join("results", "audit", "fov_crop.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "train_half_extent_mm"}, indent=1),
          {k: v for k, v in out["train_half_extent_mm"].items() if k != "per_case"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="prostate", choices=["prostate", "prostate_ms"])
    main_ms() if ap.parse_args().task == "prostate_ms" else main()
