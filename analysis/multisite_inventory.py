"""
Multi-site prostate (raw data/MultiSiteProstate/<site>/CaseNN.nii.gz + segmentation): inventory and leakage audit
(master plan item 6, 18 Oct 2026). CPU only; reads images and masks only, trains nothing. Memory-lean: each volume
is read once and only a 96 x 96-per-slice float16 copy is kept; candidate pairs are re-read at full resolution.

Inventory, per site: cases, volume shape (z, y, x), slice count, voxel spacing (mm), mask label values, gland slices.
Leakage audit of every multi-site case against
  - our PROMISE12 clients (raw data/PROMISE12/CaseNN.mhd, all 50 training cases),
  - our Decathlon Task05 server data (imagesTr, T2 = channel 0) and Decathlon imagesTs (not used by us),
  - every case of the OTHER multi-site sites (cross-site duplicates),
in three steps (all arrays read with SimpleITK as (z, y, x)):
  1. exact: SHA-256 of the image voxel array (float32): byte-identical copies;
  2. screen: for every pair with the same slice count, both volumes resized in-plane to 96 x 96, then the best
     Pearson r over the 8 in-plane flips/transposes x z-flip; pairs with r >= 0.9 are candidates;
  3. confirm: candidates are re-read at full resolution (the reference resized to the multi-site in-plane grid if
     the grids differ), r recomputed under the screened alignment, and the mask Dice computed.
     DUPLICATE if full-resolution r >= 0.99 on the same grid or >= 0.95 after resizing.
Writes results/audit/multisite_inventory.json and reports/audit/multisite_inventory.md.
    & $py analysis/multisite_inventory.py
"""
import glob
import hashlib
import json
import os
import re
import sys

import cv2
import numpy as np
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
MS = os.path.join("raw data", "MultiSiteProstate")
P12 = os.path.join("raw data", "PROMISE12")
DEC = os.path.join("raw data", "Task05_Prostate", "Task05_Prostate")
SMALL = 96
SCREEN_R = 0.9
DUP_R, DUP_R_RESAMPLED = 0.99, 0.95


def read(path):
    img = sitk.ReadImage(path)
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()
    if a.ndim == 4:                       # Decathlon (t, z, y, x): T2 = channel 0
        a, sp = a[0], sp[:3]
    return a, tuple(round(float(s), 4) for s in sp)  # spacing (x, y, z)


def mask_path(p):
    for cand in [p.replace(".nii.gz", "_segmentation.nii.gz"), p.replace(".nii.gz", "_Segmentation.nii.gz"),
                 p.replace(".mhd", "_segmentation.mhd"), p.replace("imagesTr", "labelsTr")]:
        if cand != p and os.path.exists(cand):
            return cand
    return None


def resize_vol(a, shape, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return np.stack([cv2.resize(sl, (shape[1], shape[0]), interpolation=interp) for sl in a.astype(np.float32)])


def variants(a):
    for zf in (False, True):
        b = a[::-1] if zf else a
        for t in (False, True):
            c = np.swapaxes(b, 1, 2) if t else b
            for fy in (False, True):
                for fx in (False, True):
                    d = c[:, ::-1] if fy else c
                    yield (zf, t, fy, fx), (d[:, :, ::-1] if fx else d)


def apply(a, key):
    zf, t, fy, fx = key
    b = a[::-1] if zf else a
    b = np.swapaxes(b, 1, 2) if t else b
    b = b[:, ::-1] if fy else b
    return b[:, :, ::-1] if fx else b


def corr(x, y):
    x, y = x.ravel().astype(np.float64), y.ravel().astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else 0.0


def best_align(a, b):
    best = (-2.0, None)
    for key, v in variants(a):
        if v.shape == b.shape:
            r = corr(v, b)
            if r > best[0]:
                best = (r, key)
    return best


def dice(a, b):
    a, b = a > 0, b > 0
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def describe(p, tag):
    a, sp = read(p)
    mp = mask_path(p)
    m = read(mp)[0] if mp else None
    d = {"set": tag, "path": p, "case": os.path.basename(p).split(".")[0], "shape": list(a.shape), "spacing": list(sp),
         "sha": hashlib.sha256(a.tobytes()).hexdigest(), "mask": mp,
         "small": resize_vol(a, (SMALL, SMALL)).astype(np.float16)}
    if m is not None:
        d["labels"] = sorted(int(v) for v in np.unique(m))
        d["gland_slices"] = int((m > 0).any(axis=(1, 2)).sum())
    return d


def confirm(c, r, key):
    a = read(c["path"])[0]
    b = read(r["path"])[0]
    resampled = sorted(a.shape[1:]) != sorted(b.shape[1:])
    av = apply(a, key)
    if av.shape[1:] != b.shape[1:]:
        b = resize_vol(b, av.shape[1:])
    rr = corr(av, b)
    md = None
    if c["mask"] and r["mask"]:
        am, bm = apply(read(c["mask"])[0], key), read(r["mask"])[0]
        if am.shape[1:] != bm.shape[1:]:
            bm = resize_vol(bm, am.shape[1:], nearest=True)
        md = dice(am, bm)
    return {"r_full": round(rr, 5), "resampled": resampled, "mask_dice": None if md is None else round(md, 4),
            "duplicate": rr >= (DUP_R_RESAMPLED if resampled else DUP_R)}


def main():
    sites = sorted(d for d in os.listdir(MS) if os.path.isdir(os.path.join(MS, d)))
    ms = {s: [describe(p, s) for p in sorted(glob.glob(os.path.join(MS, s, "Case*.nii.gz")))
              if not re.search(r"_[Ss]egmentation", p)] for s in sites}
    dec = lambda sub: sorted(p for p in glob.glob(os.path.join(DEC, sub, "*.nii.gz")) if not os.path.basename(p).startswith("._"))
    refs = [describe(p, "PROMISE12") for p in sorted(glob.glob(os.path.join(P12, "Case[0-9][0-9].mhd")))] + \
        [describe(p, "Decathlon_imagesTr") for p in dec("imagesTr")] + [describe(p, "Decathlon_imagesTs") for p in dec("imagesTs")]
    allms = [c for cs in ms.values() for c in cs]

    def screen_and_confirm(c, pool):
        found = []
        for r in pool:
            if r is c or r["shape"][0] != c["shape"][0]:
                continue
            exact = r["sha"] == c["sha"]
            rs, key = best_align(c["small"].astype(np.float32), r["small"].astype(np.float32))
            if exact or (key is not None and rs >= SCREEN_R):
                found.append({"ref": f"{r['set']}/{r['case']}", "exact_hash": exact, "r_screen": round(rs, 4),
                              **confirm(c, r, key or (False, False, False, False))})
        return found

    inventory, audit, cross = {}, {}, []
    for s, cases in ms.items():
        inventory[s] = {
            "cases": len(cases), "missing_masks": [c["case"] for c in cases if not c["mask"]],
            "slices": {"min": min(c["shape"][0] for c in cases), "max": max(c["shape"][0] for c in cases),
                       "total": sum(c["shape"][0] for c in cases)},
            "slices_with_gland": sum(c.get("gland_slices", 0) for c in cases),
            "inplane_shapes": sorted({f"{c['shape'][1]}x{c['shape'][2]}" for c in cases}),
            "spacing_xy_mm": sorted({f"{c['spacing'][0]:.3f}x{c['spacing'][1]:.3f}" for c in cases}),
            "spacing_z_mm": sorted({round(c["spacing"][2], 2) for c in cases}),
            "mask_labels": sorted({v for c in cases for v in c.get("labels", [])})}
        rows = []
        for c in cases:
            m = screen_and_confirm(c, refs)
            x = screen_and_confirm(c, [o for o in allms if o["set"] != s])
            cross += [{"a": f"{s}/{c['case']}", **y} for y in x]
            rows.append({"case": c["case"], "matches": m,
                         "duplicate_of": sorted(y["ref"] for y in m if y["duplicate"] or y["exact_hash"])})
        audit[s] = {"cases": rows, "n_duplicate_cases": sum(bool(r["duplicate_of"]) for r in rows),
                    "duplicate_of_sets": sorted({d.split("/")[0] for r in rows for d in r["duplicate_of"]})}
    cross_dups = [x for x in cross if x["duplicate"] or x["exact_hash"]]
    for s in audit:
        audit[s]["cross_site_duplicates"] = sorted({x["ref"] for x in cross_dups if x["a"].startswith(s + "/")})
        audit[s]["clean_vs_promise12_and_decathlon"] = audit[s]["n_duplicate_cases"] == 0
    refs_inv = {t: sum(r["set"] == t for r in refs) for t in ["PROMISE12", "Decathlon_imagesTr", "Decathlon_imagesTs"]}
    out = {"inventory": inventory, "audit": audit, "cross_site_candidates": cross, "reference_sets": refs_inv,
           "thresholds": {"screen_r_96px": SCREEN_R, "duplicate_r_same_grid": DUP_R, "duplicate_r_resampled": DUP_R_RESAMPLED}}
    os.makedirs(os.path.join("results", "audit"), exist_ok=True)
    with open(os.path.join("results", "audit", "multisite_inventory.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    L = ["# Multi-site prostate: inventory and leakage audit", "",
         "Source: `analysis/multisite_inventory.py` → `results/audit/multisite_inventory.json`. CPU only, no training.", "",
         "| Site | Cases | Slices (min–max; total) | Slices with gland | In-plane size | Spacing x×y (mm) | Spacing z (mm) | Mask labels |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in inventory.items():
        xy, z = v["spacing_xy_mm"], v["spacing_z_mm"]
        L.append(f"| {s} | {v['cases']} | {v['slices']['min']}–{v['slices']['max']}; {v['slices']['total']} | "
                 f"{v['slices_with_gland']} | {', '.join(v['inplane_shapes'])} | {', '.join(xy[:3])}{' …' if len(xy) > 3 else ''} | "
                 f"{', '.join(map(str, z[:4]))}{' …' if len(z) > 4 else ''} | {v['mask_labels']} |")
    L += ["", f"References: PROMISE12 {refs_inv['PROMISE12']} cases (our clients); Decathlon imagesTr "
          f"{refs_inv['Decathlon_imagesTr']} (our server sets); Decathlon imagesTs {refs_inv['Decathlon_imagesTs']} (unused).", "",
          "| Site | Cases duplicating PROMISE12 / Decathlon | Of which sets | Cross-site duplicates | Clean vs PROMISE12 + Decathlon |",
          "|---|---|---|---|---|"]
    for s, a in audit.items():
        L.append(f"| {s} | {a['n_duplicate_cases']} of {inventory[s]['cases']} | {', '.join(a['duplicate_of_sets']) or '—'} | "
                 f"{len(a['cross_site_duplicates'])} | {'yes' if a['clean_vs_promise12_and_decathlon'] else 'no'} |")
    L += ["", "Per-case duplicates (case → reference: full-resolution r, mask Dice):", ""]
    for s, a in audit.items():
        for r in a["cases"]:
            ms_ = [m for m in r["matches"] if m["duplicate"] or m["exact_hash"]]
            if ms_:
                L.append(f"- {s}/{r['case']} → " + "; ".join(f"{m['ref']} (r={m['r_full']}, Dice={m['mask_dice']}"
                                                             f"{', exact hash' if m['exact_hash'] else ''})" for m in ms_))
    if cross_dups:
        L += ["", "Cross-site duplicates:", ""] + [f"- {x['a']} = {x['ref']} (r={x['r_full']}, Dice={x['mask_dice']})"
                                                  for x in cross_dups]
    nd = [m for a in audit.values() for r in a["cases"] for m in r["matches"] if not (m["duplicate"] or m["exact_hash"])]
    L += ["", f"Screened candidates (r ≥ {SCREEN_R} at 96 px) that were NOT confirmed as duplicates: {len(nd)}"
          + (f" (max full-resolution r {max(m['r_full'] for m in nd):.3f})" if nd else "") + "."]
    os.makedirs(os.path.join("reports", "audit"), exist_ok=True)
    with open(os.path.join("reports", "audit", "multisite_inventory.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
