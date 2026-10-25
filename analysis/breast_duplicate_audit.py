"""
Breast duplicate / leakage audit (18 Oct 2026, after the prostate leakage finding). CPU only, images only.

Data: BUSI and BUS-UCLM (our clients) and BUS-BRA (our server: fitness / validation / test, split by patient).
Checks:
  - BUSI vs BUS-BRA, BUS-UCLM vs BUS-BRA, BUSI vs BUS-UCLM (cross-dataset duplicates);
  - inside BUS-BRA: pairs of images from DIFFERENT patients (bus_data.csv `Case`), and whether they cross the
    fitness / validation / test split (patient roles from the run split logs);
  - inside BUSI and inside BUS-UCLM (reported; a duplicate inside one client is not a train/server leak).
Method (memory-lean): every image read once as grayscale and kept as a 64 x 64 z-normalised copy.
  1. exact: SHA-256 of the decoded pixel array;
  2. screen: Pearson r of every pair (one matrix product), also against the horizontally flipped image; r >= 0.95;
  3. confirm: screened pairs re-read and compared at 256 x 256: DUPLICATE if r >= 0.98, NEAR-DUPLICATE if 0.95-0.98.
  Limitation: whole-image correlation does not detect a copy that was cropped or padded differently.
Writes results/audit/breast_duplicates.json and docs/protocol/breast_duplicates.md.
    & $py analysis/breast_duplicate_audit.py
"""
import csv
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from data.breast_dataset import INDEXERS  # noqa: E402
from experiments.config import load_config  # noqa: E402

SMALL, BIG = 64, 256
SCREEN, DUP, NEAR = 0.95, 0.98, 0.95


def gray(p, size):
    a = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return a, cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def znorm(v):
    v = v.reshape(len(v), -1).astype(np.float32)
    v = v - v.mean(1, keepdims=True)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)


def main():
    cfg = load_config("breast")
    roots = {c["name"]: c["path"] for c in cfg["clients"]}
    roots[cfg["server"]["name"]] = cfg["server"]["path"]
    items = []
    for ds, root in roots.items():
        for img, _ in INDEXERS[ds](root):
            items.append({"ds": ds, "path": img, "id": os.path.splitext(os.path.basename(img))[0]})
    small, sha = [], []
    for it in items:
        full, s = gray(it["path"], SMALL)
        small.append(s)
        sha.append(hashlib.sha256(full.tobytes()).hexdigest())
    X = znorm(np.stack(small))
    Xf = znorm(np.stack([s[:, ::-1] for s in small]))
    S = np.maximum(X @ X.T, X @ Xf.T)
    np.fill_diagonal(S, -1)
    meta = {r["ID"]: r["Case"] for r in csv.DictReader(open(cfg["server"]["metadata"] if "metadata" in cfg["server"] else
                                                            os.path.join(roots["BUS-BRA"], "BUSBRA", "bus_data.csv"),
                                                            encoding="utf-8"))}
    logs = sorted(glob.glob(os.path.join("results", "runs", "breast_*_seed*_????????", "split_log.json")))
    srv = json.load(open(logs[0], encoding="utf-8"))["server"]
    role_of = {p: r for r, v in srv.items() for p in v["patients"]}
    ii, jj = np.where(np.triu(S, 1) >= SCREEN)
    by_sha = {}
    for k, h in enumerate(sha):
        by_sha.setdefault(h, []).append(k)
    exact = {tuple(sorted((a, b))) for ks in by_sha.values() for x, a in enumerate(ks) for b in ks[x + 1:]}
    cand = sorted(set(zip(ii.tolist(), jj.tolist())) | exact)
    pairs = []
    for a, b in cand:
        A, B = gray(items[a]["path"], BIG)[1], gray(items[b]["path"], BIG)[1]
        r = max(float(znorm(A[None])[0] @ znorm(B[None])[0]), float(znorm(A[None])[0] @ znorm(B[None, :, ::-1].copy())[0]))
        ia, ib = items[a], items[b]
        rec = {"a": f"{ia['ds']}/{ia['id']}", "b": f"{ib['ds']}/{ib['id']}", "r_screen": round(float(S[a, b]), 4),
               "r_256": round(r, 4), "exact": (a, b) in exact,
               "kind": "duplicate" if (a, b) in exact or r >= DUP else ("near-duplicate" if r >= NEAR else "not confirmed")}
        if ia["ds"] == ib["ds"] == "BUS-BRA":
            pa, pb = meta.get(ia["id"]), meta.get(ib["id"])
            rec.update({"patients": [pa, pb], "roles": [role_of.get(pa, "excluded"), role_of.get(pb, "excluded")]})
        pairs.append(rec)

    def group(p):
        da, db = p["a"].split("/")[0], p["b"].split("/")[0]
        if da != db:
            return " vs ".join(sorted([da, db]))
        if da == "BUS-BRA":
            if p["patients"][0] == p["patients"][1]:
                return "BUS-BRA same patient"
            return "BUS-BRA different patients, " + ("across splits" if p["roles"][0] != p["roles"][1] else "same split")
        return f"inside {da}"
    summary = {}
    for p in pairs:
        g = summary.setdefault(group(p), {"duplicate": 0, "near-duplicate": 0, "not confirmed": 0})
        g[p["kind"]] += 1
    leak_groups = [g for g in summary if " vs " in g or "across splits" in g]
    leaks = {g: summary[g] for g in leak_groups if summary[g]["duplicate"] or summary[g]["near-duplicate"]}
    out = {"n_images": {ds: sum(it["ds"] == ds for it in items) for ds in roots}, "thresholds":
           {"screen_r_64px": SCREEN, "duplicate_r_256px": DUP, "near_duplicate_r_256px": NEAR},
           "summary": summary, "leak_groups_with_findings": leaks, "pairs": pairs}
    os.makedirs(os.path.join("results", "audit"), exist_ok=True)
    with open(os.path.join("results", "audit", "breast_duplicates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    L = ["# Breast duplicate / leakage audit", "",
         "Source: `analysis/breast_duplicate_audit.py` → `results/audit/breast_duplicates.json`. CPU only, images only.", "",
         "Images: " + ", ".join(f"{k} {v}" for k, v in out["n_images"].items()) + ".", "",
         "| Pair group | Duplicates (r ≥ 0.98 at 256 px or identical pixels) | Near-duplicates (0.95–0.98) | Screened, not confirmed |",
         "|---|---|---|---|"]
    order = ["BUS-BRA vs BUSI", "BUS-BRA vs BUS-UCLM", "BUS-UCLM vs BUSI", "BUS-BRA different patients, across splits",
             "BUS-BRA different patients, same split", "BUS-BRA same patient", "inside BUSI", "inside BUS-UCLM"]
    for g in order + sorted(k for k in summary if k not in order):
        v = summary.get(g, {"duplicate": 0, "near-duplicate": 0, "not confirmed": 0})
        L.append(f"| {g} | {v['duplicate']} | {v['near-duplicate']} | {v['not confirmed']} |")
    L += ["", "**Train/server or cross-split leakage found: " + ("YES — " + ", ".join(leaks) if leaks else "none") + ".**",
          "", "Limitation: a copy cropped or padded differently is not detected by whole-image correlation."]
    if leaks:
        L += ["", "Leaking pairs:", ""] + [f"- {p['a']} = {p['b']} (r256={p['r_256']}, {p['kind']}"
                                         + (f", patients {p['patients']}, roles {p['roles']}" if "roles" in p else "") + ")"
                                         for p in pairs if group(p) in leaks and p["kind"] != "not confirmed"][:200]
    os.makedirs(os.path.join("docs", "protocol"), exist_ok=True)
    with open(os.path.join("docs", "protocol", "breast_duplicates.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
