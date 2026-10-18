"""
Three-way server splits (fitness / validation / test) at patient or case level,
plus the fixed subsamples used by the GA fitness and by q_i.

All splits are deterministic given the seed in the task YAML and are written to
results/splits/<task>.json (case/patient ids, image or slice counts) by
`write_split_log`. Assumptions (approved 15 Oct 2026, plan.md section 2, Q1):
  - BUS-BRA: patients (bus_data.csv column `Case`) are split 15 / 25 / 60 %
    into fitness / validation / test, stratified by (device, pathology).
    Images of one patient never cross sets.
  - Decathlon Task05: the 32 labelled cases are split 8 / 8 / 16. Cases are
    sorted by number of non-empty slices (gland-size proxy) and taken in
    blocks of 4 similar cases; each block contributes 1 fitness, 1 validation
    and 2 test cases, the roles within a block drawn with the seed.
  - Fixed fitness subsample: `n` server images chosen once from the fitness
    set and shared by every GA method (FedGA (baseline) and XFedGA):
    one image per patient (breast) or evenly spaced slices per case (prostate).
  - Fixed q_i subsample: `n` training images per client, chosen once.
"""
import csv
import json
import os
from collections import defaultdict

import numpy as np

ROLES = ("fitness", "validation", "test")


def _stratified_groups(groups, strata, fractions, seed):
    """Assign group ids to ROLES with per-stratum proportions `fractions` (fitness, validation)."""
    rng = np.random.RandomState(seed)
    out = {r: [] for r in ROLES}
    by_stratum = defaultdict(list)
    for g in sorted(groups):
        by_stratum[strata[g]].append(g)
    for key in sorted(by_stratum):
        members = [by_stratum[key][i] for i in rng.permutation(len(by_stratum[key]))]
        n = len(members)
        n_fit = int(round(fractions[0] * n))
        n_val = int(round(fractions[1] * n))
        out["fitness"] += members[:n_fit]
        out["validation"] += members[n_fit:n_fit + n_val]
        out["test"] += members[n_fit + n_val:]
    return {r: sorted(v) for r, v in out.items()}


def busbra_patient_split(pairs, metadata_csv, fractions=(0.15, 0.25), seed=42):
    """
    pairs: BUS-BRA (image, mask) pairs (already filtered for non-empty masks).
    Returns {role: [pairs]}, {role: [patient ids]}, {image id: patient id}.
    """
    meta = {r["ID"]: r for r in csv.DictReader(open(metadata_csv, encoding="utf-8"))}
    img_id = {p[0]: os.path.splitext(os.path.basename(p[0]))[0] for p in pairs}
    patient_of = {img_id[p[0]]: meta[img_id[p[0]]]["Case"] for p in pairs}
    strata = {}
    for p in pairs:
        row = meta[img_id[p[0]]]
        strata.setdefault(row["Case"], (row["Device"], row["Pathology"]))
    patients = _stratified_groups(set(patient_of.values()), strata, fractions, seed)
    role_of = {pid: r for r, pids in patients.items() for pid in pids}
    split = {r: [] for r in ROLES}
    for p in pairs:
        split[role_of[patient_of[img_id[p[0]]]]].append(p)
    return split, patients, patient_of


def decathlon_case_split(cases, slices_per_case, seed=42, block_roles=("fitness", "validation", "test", "test"),
                         remainder_roles=None):
    """
    cases: list of (case_path, tag); slices_per_case: {case_path: n non-empty slices}.
    Cases sorted by gland size (slice count) fill blocks of len(block_roles); each block's roles are permuted.
    remainder_roles: roles for the cases left after the last full block (the largest glands), assigned by a seeded
    permutation drawn after the blocks (the block assignment is unchanged); required if cases remain.
    Returns {role: [cases]}.
    """
    order = sorted(cases, key=lambda c: (slices_per_case[c[0]], os.path.basename(c[0])))
    rem = len(order) % len(block_roles)
    if rem and (remainder_roles is None or len(remainder_roles) != rem):
        raise ValueError(f"{len(order)} cases do not fill blocks of {len(block_roles)}; give {rem} remainder_roles")
    rng = np.random.RandomState(seed)
    split = {r: [] for r in ROLES}
    full = len(order) - rem
    for b in range(0, full, len(block_roles)):
        block = order[b:b + len(block_roles)]
        for case, k in zip(block, rng.permutation(len(block_roles))):
            split[block_roles[k]].append(case)
    if rem:
        for case, k in zip(order[full:], rng.permutation(rem)):
            split[remainder_roles[k]].append(case)
    return {r: sorted(v) for r, v in split.items()}


def group_split(items, group_of, val_fraction=0.2, seed=42):
    """Client train/validation split by group (e.g. BUS-UCLM patient prefix). Returns (train, val)."""
    groups = sorted({group_of(i) for i in items})
    rng = np.random.RandomState(seed)
    n_val = int(round(val_fraction * len(groups)))
    val_groups = {groups[i] for i in rng.permutation(len(groups))[:n_val]}
    train = [i for i in items if group_of(i) not in val_groups]
    val = [i for i in items if group_of(i) in val_groups]
    return train, val


def one_per_group_subsample(items, group_of, n, seed=42):
    """Fixed subsample: `n` items from `n` different groups (e.g. one image per patient)."""
    rng = np.random.RandomState(seed)
    by_group = defaultdict(list)
    for i, item in enumerate(items):
        by_group[group_of(item)].append(i)
    groups = sorted(by_group)
    if n > len(groups):
        raise ValueError(f"need {n} groups, have {len(groups)}")
    chosen = [groups[i] for i in rng.permutation(len(groups))[:n]]
    return sorted(by_group[g][rng.randint(len(by_group[g]))] for g in chosen)


def evenly_spaced_per_group(group_sizes, n):
    """
    Fixed subsample for slice data: `group_sizes` is a list of slice counts per case in dataset
    order; returns `n` dataset indices spread evenly over cases and, within a case, evenly over
    its slices (deterministic, no seed needed).
    """
    if n >= sum(group_sizes):          # the subsample would cover the whole set: use every slice
        return list(range(sum(group_sizes)))
    k = len(group_sizes)
    per = [n // k + (1 if i < n % k else 0) for i in range(k)]
    out, offset = [], 0
    for size, m in zip(group_sizes, per):
        if m > size:
            raise ValueError(f"case has {size} slices, need {m}")
        out += [offset + int(j) for j in np.linspace(0, size - 1, m + 2)[1:-1].round()]
        offset += size
    return out


def random_subsample(n_items, n, seed=42):
    """Fixed subsample of `n` indices out of `n_items` (q_i images per client)."""
    if n >= n_items:
        return list(range(n_items))
    return sorted(np.random.RandomState(seed).permutation(n_items)[:n].tolist())


def write_split_log(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
