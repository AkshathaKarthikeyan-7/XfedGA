"""
Leakage check of the MAIN prostate setting (18 Oct 2026; triggered by analysis/multisite_inventory.py, which found
RUNMC cases duplicated in both PROMISE12 and Decathlon). CPU only, images and masks only, no results read.

Compares every PROMISE12 training case (our clients) with every Decathlon Task05 case (imagesTr = our server
fitness / validation / test; imagesTs unused) with the same method as multisite_inventory.py (96-px screen over
the 16 alignments, full-resolution confirmation with the reference resized to the PROMISE12 grid; duplicate if
r >= 0.95 after resizing / 0.99 on the same grid). For each duplicate, reports the PROMISE12 client it belongs to
(random partition, data seed 42 as in every run) and the Decathlon server role (fitness / validation / test) from
the split logs. Writes results/audit/promise12_decathlon_overlap.json and reports/audit/promise12_decathlon_overlap.md.
    & $py analysis/promise12_decathlon_overlap.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multisite_inventory import DEC, P12, SCREEN_R, best_align, confirm, describe  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)


def roles():
    """Client of each PROMISE12 case and role of each Decathlon case, from the split logs of the runs."""
    logs = sorted(glob.glob(os.path.join("results", "runs", "prostate_*_seed*_????????", "split_log.json")))
    def cases(p):                     # case assignment only (q_i subsamples are seed-dependent)
        s = json.load(open(p, encoding="utf-8"))
        return json.dumps({part: {k: sorted(v["cases"]) for k, v in s[part].items()} for part in ("clients", "server")},
                          sort_keys=True)
    sets = {cases(p) for p in logs}
    assert len(sets) == 1, "the untagged prostate runs do not share one case assignment"
    log = json.loads(sets.pop())
    server = {c: role for role, v in log["server"].items() for c in v}
    clients = {c: name for name, v in log["clients"].items() for c in v}
    return clients, server


def main():
    p12 = [describe(p, "PROMISE12") for p in sorted(glob.glob(os.path.join(P12, "Case[0-9][0-9].mhd")))]
    dec = [describe(p, f"Decathlon_{sub}") for sub in ("imagesTr", "imagesTs")
           for p in sorted(glob.glob(os.path.join(DEC, sub, "*.nii.gz"))) if not os.path.basename(p).startswith("._")]
    pairs = []
    for c in p12:
        for r in dec:
            if r["shape"][0] != c["shape"][0]:
                continue
            rs, key = best_align(c["small"].astype(np.float32), r["small"].astype(np.float32))
            if key is not None and rs >= SCREEN_R:
                res = confirm(c, r, key)
                pairs.append({"promise12": c["case"], "decathlon": f"{r['set']}/{r['case']}", "r_screen": round(rs, 4), **res})
    dups = [p for p in pairs if p["duplicate"]]
    clients, server = roles()
    for p in dups:
        p["client"] = clients.get(p["promise12"], "unknown")
        name = p["decathlon"].split("/")[1]
        p["server_role"] = server.get(name, "not used (imagesTs)" if "imagesTs" in p["decathlon"] else "unknown")
    by_role = {}
    for p in dups:
        by_role[p["server_role"]] = sorted(set(by_role.get(p["server_role"], [])) | {p["decathlon"].split("/")[1]})
    n_server = {role: len([c for c, r_ in server.items() if r_ == role]) for role in set(server.values())}
    out = {"pairs_screened_ge_r": SCREEN_R, "candidates": pairs, "duplicates": dups,
           "duplicated_promise12_cases": sorted({p["promise12"] for p in dups}),
           "duplicated_decathlon_cases_by_role": by_role, "server_role_sizes": n_server,
           "clients_of_duplicated_cases": {k: sorted({p["promise12"] for p in dups if p["client"] == k})
                                           for k in sorted({p["client"] for p in dups})}}
    os.makedirs(os.path.join("results", "audit"), exist_ok=True)
    with open(os.path.join("results", "audit", "promise12_decathlon_overlap.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    L = ["# PROMISE12 (clients) vs Decathlon Task05 (server): duplicate cases", "",
         "Source: `analysis/promise12_decathlon_overlap.py` → `results/audit/promise12_decathlon_overlap.json`. "
         "Images and masks only; no results read.", "",
         f"Duplicates: **{len(dups)}** pairs; PROMISE12 cases involved: {len(out['duplicated_promise12_cases'])} of {len(p12)}.", "",
         "| Decathlon server role | Cases in role | Duplicated by a PROMISE12 training case | Cases |", "|---|---|---|---|"]
    for role in ["fitness", "validation", "test"] + sorted(k for k in by_role if k not in ("fitness", "validation", "test")):
        L.append(f"| {role} | {n_server.get(role, '—')} | {len(by_role.get(role, []))} | {', '.join(by_role.get(role, [])) or '—'} |")
    L += ["", "| PROMISE12 case | Client | Decathlon case | Role | r (full res.) | Mask Dice |", "|---|---|---|---|---|---|"]
    for p in sorted(dups, key=lambda p: p["promise12"]):
        L.append(f"| {p['promise12']} | {p['client']} | {p['decathlon']} | {p['server_role']} | {p['r_full']} | {p['mask_dice']} |")
    nd = [p for p in pairs if not p["duplicate"]]
    L += ["", f"Screened candidates not confirmed: {len(nd)}."]
    os.makedirs(os.path.join("reports", "audit"), exist_ok=True)
    with open(os.path.join("reports", "audit", "promise12_decathlon_overlap.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
