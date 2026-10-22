"""
Build the data side of one task from its config: client training sets, the fixed
q_i subsamples, and the server fitness / validation / test sets with the fixed
fitness subsample. Every server sample carries an image id and a group id
(patient for BUS-BRA, case for Decathlon) so test metrics can be clustered.

Splits depend only on the data seed (`seed` in the YAML), never on the run seed,
so every method and every run seed sees identical splits and subsamples.
"""
import os
from dataclasses import dataclass, field

from data.breast_dataset import (INDEXERS as BREAST_INDEXERS, BreastClientDataset, breast_cache,
                                 filter_nonempty_masks, train_val_split)
from data.prostate_dataset import INDEXERS as PROSTATE_INDEXERS, ProstateClientDataset, prostate_cache, \
    split_promise12_clients
from data.scenarios import ShuffledMaskDataset, split_promise12_by_acquisition
from data.splits import (ROLES, busbra_patient_split, decathlon_case_split, evenly_spaced_per_group,
                         group_split, one_per_group_subsample, random_subsample)


@dataclass
class ClientData:
    name: str
    train: object            # torch Dataset
    qi_indices: list         # fixed q_i subsample (indices into train)
    n_train_groups: int      # patients / cases (or images where no patient id exists)


@dataclass
class ServerSet:
    dataset: object
    image_ids: list
    group_ids: list


@dataclass
class TaskData:
    clients: list
    server: dict                         # role -> ServerSet
    fitness_subsample: list              # indices into server["fitness"].dataset
    val_explain_subsample: list          # indices into server["validation"].dataset
    split_log: dict = field(default_factory=dict)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def build_breast(cfg, part="all"):
    size, seed = tuple(cfg["image_size"]), cfg["seed"]
    sp = cfg["splits"]
    clients, log = [], {"clients": {}, "server": {}}
    for k, c in enumerate(cfg["clients"]):
        if part == "server" or (isinstance(part, int) and part != k):
            continue
        pairs = filter_nonempty_masks(BREAST_INDEXERS[c["name"]](c["path"]), size=size, verbose=False,
                                      cache=breast_cache(c["name"], c["path"], size))
        if c["name"] == "BUS-UCLM":   # patient = filename prefix, e.g. ALWI_000.png -> ALWI
            train, val = group_split(pairs, lambda p: _stem(p[0]).split("_")[0], sp["client_val_fraction"], seed)
            groups = len({_stem(p[0]).split("_")[0] for p in train})
        else:                         # BUSI has no patient ids: image-level split (stated limitation)
            train, val = train_val_split(pairs, val_ratio=sp["client_val_fraction"], seed=seed)
            groups = len(train)
        ds = BreastClientDataset(c["name"], c["path"], size=size, pairs=train)
        qi = random_subsample(len(ds), cfg["saliency"]["subsample"], seed=seed + 1000 + k)
        clients.append(ClientData(c["name"], ds, qi, groups))
        log["clients"][c["name"]] = {"n_train_images": len(train), "n_heldout_images": len(val),
                                     "n_train_groups": groups, "qi_subsample": [_stem(train[i][0]) for i in qi]}
    if isinstance(part, int):
        return TaskData(clients, {}, [], [], log)

    s = cfg["server"]
    pairs = filter_nonempty_masks(BREAST_INDEXERS[s["name"]](s["path"]), size=size, verbose=False,
                                  cache=breast_cache(s["name"], s["path"], size))
    fr = sp["server_fractions"]
    split, patients, patient_of = busbra_patient_split(pairs, sp["server_metadata"],
                                                       (fr["fitness"], fr["validation"]), seed)
    server = {}
    for role in ROLES:
        ids = [_stem(p[0]) for p in split[role]]
        server[role] = ServerSet(BreastClientDataset(s["name"], s["path"], size=size, pairs=split[role]),
                                 ids, [patient_of[i] for i in ids])
        log["server"][role] = {"n_patients": len(patients[role]), "n_images": len(ids), "patients": patients[role]}
    fit_idx = one_per_group_subsample(server["fitness"].image_ids, lambda i: patient_of[i],
                                      cfg["fitness"]["subsample"], seed)
    val_idx = one_per_group_subsample(server["validation"].image_ids, lambda i: patient_of[i],
                                      cfg["evaluation"]["explain_samples"], seed + 1)
    log["fitness_subsample"] = [server["fitness"].image_ids[i] for i in fit_idx]
    log["val_explain_subsample"] = [server["validation"].image_ids[i] for i in val_idx]
    return TaskData(clients, server, fit_idx, val_idx, log)


def _case_ids(cases, cache):
    image_ids, group_ids, sizes = [], [], []
    for case_path, _ in cases:
        n = len(cache.rows(case_path))
        name = _stem(case_path).replace(".nii", "")
        image_ids += [f"{name}#{k}" for k in range(n)]
        group_ids += [name] * n
        sizes.append(n)
    return image_ids, group_ids, sizes


def build_prostate(cfg, part="all"):
    size, seed = tuple(cfg["image_size"]), cfg["seed"]
    root = cfg["clients"][0]["path"]
    clients, log = [], {"clients": {}, "server": {}}
    pp = cfg.get("preprocessing") or {}
    pre = {"fov_crop_mm": pp.get("fov_crop_mm"), "intensity": pp.get("intensity", "minmax")}
    p_cache = prostate_cache("PROMISE12", root, size, **pre)
    sc = cfg["scenario"]
    if sc["partition"] == "acquisition":
        client_cases = split_promise12_by_acquisition(root, sc["acquisition_threshold_mm"])
    else:
        client_cases = split_promise12_clients(root, seed=seed)
    log["partition"] = sc["partition"]
    for k, cases in enumerate(client_cases):
        if part == "server" or (isinstance(part, int) and part != k):
            continue
        ds = ProstateClientDataset("PROMISE12", root, size=size, cases=cases, **pre)
        qi = random_subsample(len(ds), cfg["saliency"]["subsample"], seed=seed + 1000 + k)
        ids, _, _ = _case_ids(cases, p_cache)
        name = f"PROMISE12_client{k + 1}"
        clients.append(ClientData(name, ds, qi, len(cases)))
        log["clients"][name] = {"n_cases": len(cases), "n_slices": len(ds),
                                "cases": sorted(_stem(c) for c, _ in cases), "qi_subsample": [ids[i] for i in qi]}
    if isinstance(part, int):
        return TaskData(clients, {}, [], [], log)

    s = cfg["server"]
    cases = PROSTATE_INDEXERS["Decathlon"](s["path"])
    d_cache = prostate_cache("Decathlon", s["path"], size, **pre)
    sizes = {c: len(d_cache.rows(c)) for c, _ in cases}
    split = decathlon_case_split(cases, sizes, seed, tuple(cfg["splits"]["server_block_roles"]))
    server, group_sizes = {}, {}
    for role in ROLES:
        ids, groups, gsz = _case_ids(split[role], d_cache)
        server[role] = ServerSet(ProstateClientDataset("Decathlon", s["path"], size=size, cases=split[role], **pre),
                                 ids, groups)
        group_sizes[role] = gsz
        log["server"][role] = {"n_cases": len(split[role]), "n_slices": len(ids),
                               "cases": sorted({g for g in groups})}
    fit_idx = evenly_spaced_per_group(group_sizes["fitness"], cfg["fitness"]["subsample"])
    val_idx = evenly_spaced_per_group(group_sizes["validation"], cfg["evaluation"]["explain_samples"])
    log["fitness_subsample"] = [server["fitness"].image_ids[i] for i in fit_idx]
    log["val_explain_subsample"] = [server["validation"].image_ids[i] for i in val_idx]
    return TaskData(clients, server, fit_idx, val_idx, log)


def build_prostate_ms(cfg, part="all"):
    """
    Multi-site prostate (the main prostate experiment from 18 Oct 2026; replaces PROMISE12 -> Decathlon, which leaked:
    reports/audit/promise12_decathlon_overlap.md). One client per site (cfg["clients"]: BIDMC, HK, I2CVB, RUNMC, UCL;
    every case of the site is client training data); server = one held-out site (BMC), split case-level into
    fitness / validation / test in gland-size blocks (splits.server_block_roles + splits.server_remainder_roles).
    No PROMISE12 or Decathlon data.
    """
    size, seed = tuple(cfg["image_size"]), cfg["seed"]
    clients, log = [], {"clients": {}, "server": {}}
    pp = cfg.get("preprocessing") or {}
    pre = {"fov_crop_mm": pp.get("fov_crop_mm"), "intensity": pp.get("intensity", "minmax")}
    log["partition"] = "site"
    for k, c in enumerate(cfg["clients"]):
        if part == "server" or (isinstance(part, int) and part != k):
            continue
        name = f"MS_{c['name']}"
        cases = PROSTATE_INDEXERS[name](c["path"])
        cache = prostate_cache(name, c["path"], size, **pre)
        ds = ProstateClientDataset(name, c["path"], size=size, cases=cases, **pre)
        qi = random_subsample(len(ds), cfg["saliency"]["subsample"], seed=seed + 1000 + k)
        ids, _, _ = _case_ids(cases, cache)
        clients.append(ClientData(c["name"], ds, qi, len(cases)))
        log["clients"][c["name"]] = {"n_cases": len(cases), "n_slices": len(ds),
                                     "cases": sorted(_stem(p).replace(".nii", "") for p, _ in cases),
                                     "qi_subsample": [ids[i] for i in qi]}
    if isinstance(part, int):
        return TaskData(clients, {}, [], [], log)
    s = cfg["server"]
    sname = f"MS_{s['name']}"
    cases = PROSTATE_INDEXERS[sname](s["path"])
    s_cache = prostate_cache(sname, s["path"], size, **pre)
    sizes = {c: len(s_cache.rows(c)) for c, _ in cases}
    sp = cfg["splits"]
    split = decathlon_case_split(cases, sizes, seed, tuple(sp["server_block_roles"]),
                                 tuple(sp["server_remainder_roles"]) if sp.get("server_remainder_roles") else None)
    server, group_sizes = {}, {}
    for role in ROLES:
        ids, groups, gsz = _case_ids(split[role], s_cache)
        server[role] = ServerSet(ProstateClientDataset(sname, s["path"], size=size, cases=split[role], **pre), ids, groups)
        group_sizes[role] = gsz
        log["server"][role] = {"n_cases": len(split[role]), "n_slices": len(ids), "cases": sorted(set(groups))}
    fit_idx = evenly_spaced_per_group(group_sizes["fitness"], cfg["fitness"]["subsample"])
    val_idx = evenly_spaced_per_group(group_sizes["validation"], cfg["evaluation"]["explain_samples"])
    log["fitness_subsample"] = [server["fitness"].image_ids[i] for i in fit_idx]
    log["val_explain_subsample"] = [server["validation"].image_ids[i] for i in val_idx]
    return TaskData(clients, server, fit_idx, val_idx, log)


def split_summary(data):
    """The run-independent part of the split log: client and server case/patient ids and counts
    (no subsamples, which depend on subsample-size settings). Identical for every run with the
    same data seed."""
    return {"clients": {k: {kk: vv for kk, vv in v.items() if kk != "qi_subsample"}
                        for k, v in data.split_log["clients"].items()},
            "server": data.split_log["server"]}


def build_task(task, cfg, part="all"):
    """
    part = "all"   : every client and the server sets (custom loop);
           "server": server fitness / validation / test sets only (fl_dist server);
           k (int) : client k only, and nothing else (fl_dist client k loads only its own data).
    """
    data = {"breast": build_breast, "prostate": build_prostate, "prostate_ms": build_prostate_ms,
            "prostate_ms_runmc": build_prostate_ms}[task](cfg, part)
    k = cfg["scenario"]["corrupted_label_client"]
    if k is not None and part in ("all", k):   # robustness scenario: client k's masks shuffled across its own images
        idx = k if part == "all" else 0
        c = data.clients[idx]
        data.clients[idx] = ClientData(c.name, ShuffledMaskDataset(c.train, cfg["seed"] + 777,
                                                                 cfg["scenario"].get("corrupted_label_fraction", 1.0)),
                                       c.qi_indices,
                                       c.n_train_groups)
        data.split_log["corrupted_label_client"] = c.name
        data.split_log["corrupted_label_fraction"] = cfg["scenario"].get("corrupted_label_fraction", 1.0)
        data.split_log["corrupted_items"] = int(len(data.clients[idx].train.corrupted))
    return data
