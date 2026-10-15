"""
Single-process federated simulation (the custom loop): FedAvg, FedGA (baseline) or XFedGA,
on the breast or prostate task, fully configured from configs/<task>_clients.yaml.

    & "D:\\conda_envs\\fedga\\python.exe" experiments/run_federated.py --task breast --method xfedga --seed 1
    ... --set aggregation.beta=1.0 train.rounds=3          (dotted config overrides)

Protocol (spec sections 2, 4, 6; plan.md section 0):
  round t: each client trains locally and (XFedGA) computes q_i on its fixed q_i subsample;
  the server aggregates (FedAvg weights, or alpha_i ~ (n_i/N) max(q_i, eps)^beta); FedGA and
  XFedGA then run the GA on the fixed fitness subsample and broadcast the best candidate
  (no post-GA blend). Every round is scored on the VALIDATION set. The checkpoint is the round
  with the best centred moving-average validation DSC. The TEST set is evaluated once, at the
  end, on that checkpoint only.

Outputs in results/runs/<run_id>/: config_resolved.yaml, round_log.csv, test_per_image.csv,
run_meta.json, split_log.json (incl. fixed subsamples) and <task>_<method>_seed<k>_valselected.pt.
The run-independent split ids are in results/splits/<task>_dataseed<seed>.json (checked on every run).
"""
import argparse
import copy
import csv
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loading import loader_kwargs  # noqa: E402
from data.splits import write_split_log  # noqa: E402
from experiments.config import config_hash, git_commit, load_config, seed_everything, torch_generator  # noqa: E402
from experiments.tasks import build_task, split_summary  # noqa: E402
from fedga.client import client_saliency_score, local_update  # noqa: E402
from fedga.evaluation import FixedSet, build_fitness_fn, per_image_metrics  # noqa: E402
from fedga.fedavg import aggregate, aggregation_weights  # noqa: E402
from fedga.ga_optimizer import run_ga  # noqa: E402
from fedga.losses import build_loss  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

log = logging.getLogger("run_federated")

# A method is a combination of switches; ablations (Phase 2) add rows here, no new logic.
METHODS = {
    "fedavg": {"aggregation": "fedavg", "search": "none", "fitness": None, "anneal": False},
    "fedga": {"aggregation": "fedavg", "search": "ga", "fitness": "dsc", "anneal": False},
    "xfedga": {"aggregation": "saliency", "search": "ga", "fitness": "multi", "anneal": True},
}


def _mean(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def select_round(val_dsc, window):
    """1-based round with the best centred moving-average validation DSC (ties -> earliest)."""
    half = window // 2
    smooth = [np.mean(val_dsc[max(0, t - half):t + half + 1]) for t in range(len(val_dsc))]
    return int(np.argmax(smooth)) + 1, smooth


def run_experiment(task, method, seed, overrides=None, out_root="results/runs", tag="", evaluate_test=True):
    seed_everything(seed)
    cfg = load_config(task, overrides)
    spec = METHODS[method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chash = config_hash({"cfg": cfg, "method": method})
    run_id = f"{task}_{method}_seed{seed}_{chash[:8]}" + (f"_{tag}" if tag else "")
    out = os.path.join(out_root, run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"task": task, "method": method, "method_switches": spec, "seed": seed, **cfg}, f, sort_keys=False)

    t0 = time.time()
    data = build_task(task, cfg)
    write_split_log(os.path.join(out, "split_log.json"), data.split_log)
    shared = os.path.join("results", "splits", f"{task}_dataseed{cfg['seed']}.json")
    summary = split_summary(data)
    if os.path.exists(shared):
        with open(shared, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(summary)):
                raise RuntimeError(f"splits differ from {shared}: data or split code changed")
    else:
        write_split_log(shared, summary)
    setup_s = time.time() - t0

    tr, ga_cfg, fit_cfg = cfg["train"], cfg["ga"], cfg["fitness"]
    lk = loader_kwargs(cfg)
    bs = tr["batch_size"]
    client_loaders = [DataLoader(c.train, batch_size=bs, shuffle=True, generator=torch_generator(seed * 100 + k), **lk)
                      for k, c in enumerate(data.clients)]
    qi_loaders = [DataLoader(Subset(c.train, c.qi_indices), batch_size=bs, shuffle=False) for c in data.clients]
    sizes = [len(c.train) for c in data.clients]
    val_loader = DataLoader(data.server["validation"].dataset, batch_size=bs, shuffle=False, **lk)
    val_explain = Subset(data.server["validation"].dataset, data.val_explain_subsample)
    fixed = FixedSet(data.server["fitness"].dataset, data.fitness_subsample, device) if spec["search"] == "ga" else None

    model = build_residual_unet().to(device)
    loss_fn = build_loss()
    rows, states = [], []
    for t in range(1, tr["rounds"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rt = {"local_s": 0.0, "qi_s": 0.0}
        t_round = time.time()
        client_states, qs = [], []
        for k, c in enumerate(data.clients):
            local = copy.deepcopy(model)
            ts = time.time()
            client_states.append(local_update(local, client_loaders[k], tr["local_epochs"], tr["lr"],
                                              tr["adam_betas"], device, loss_fn))
            rt["local_s"] += time.time() - ts
            if spec["aggregation"] == "saliency":
                ts = time.time()
                qs.append(client_saliency_score(local, qi_loaders[k], device, fit_cfg["cam_threshold"]))
                rt["qi_s"] += time.time() - ts
        agg = cfg["aggregation"]
        alpha = aggregation_weights(sizes, qs if qs else None, agg["beta"], agg["eps"])
        w_agg = aggregate(client_states, alpha)

        ts = time.time()
        history = []
        if spec["search"] == "ga":
            fitness_fn = build_fitness_fn(model, fixed, spec["fitness"], fit_cfg["weights"],
                                          fit_cfg["hd95_normalizer"], fit_cfg["cam_threshold"])
            best, history = run_ga(w_agg, fitness_fn, mode="xfedga" if spec["anneal"] else "base",
                                   population_size=ga_cfg["population"], generations=ga_cfg["generations"],
                                   pc=ga_cfg["pc"], pm=ga_cfg["pm"], sigma0=ga_cfg["sigma0"], tau=ga_cfg["tau"],
                                   cache_elite_fitness=ga_cfg["cache_elite_fitness"], return_history=True)
        else:
            best = w_agg
        rt["ga_s"] = time.time() - ts
        model.load_state_dict(best)
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        ts = time.time()
        val = per_image_metrics(model, val_loader, device, fit_cfg["hd95_normalizer"], with_cam=False)
        expl = per_image_metrics(model, DataLoader(val_explain, batch_size=bs), device, fit_cfg["hd95_normalizer"],
                                 fit_cfg["cam_threshold"], with_cam=True)
        rt["val_s"] = time.time() - ts
        row = {"round": t, "val_dsc": _mean(val, "dsc"), "val_hd95_norm": _mean(val, "hd95_norm"),
               "val_hd95_px": _mean(val, "hd95_px"), "val_ec": _mean(expl, "ec"), "val_pg": _mean(expl, "pg"),
               "q_i": json.dumps([round(q, 6) for q in qs]), "alpha_i": json.dumps([round(a, 6) for a in alpha]),
               "ga_best_per_gen": json.dumps([round(h["best"], 6) for h in history]),
               "ga_mean_per_gen": json.dumps([round(h["mean"], 6) for h in history]),
               "fitness_evals": sum(h["evaluations"] for h in history),
               **{k: round(v, 2) for k, v in rt.items()}, "round_s": round(time.time() - t_round, 2),
               "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if device.type == "cuda" else 0}
        rows.append(row)
        log.info("round %d/%d val DSC %.4f HD95n %.4f EC %.4f PG %.3f q=%s (%.0fs)", t, tr["rounds"],
                 row["val_dsc"], row["val_hd95_norm"], row["val_ec"], row["val_pg"], row["q_i"], row["round_s"])

    with open(os.path.join(out, "round_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sel_round, smooth = select_round([r["val_dsc"] for r in rows], cfg["evaluation"]["selection_smoothing"])
    ckpt = os.path.join(out, f"{task}_{method}_seed{seed}_valselected.pt")
    torch.save(states[sel_round - 1], ckpt)
    meta = {"run_id": run_id, "task": task, "method": method, "seed": seed, "config_hash": chash,
            "git_commit": git_commit(), "selected_round": sel_round, "selection": "centred moving-average validation DSC",
            "smoothed_val_dsc": smooth, "n_i": sizes, "clients": [c.name for c in data.clients],
            "server_sizes": {r: len(s.image_ids) for r, s in data.server.items()},
            "setup_s": round(setup_s, 1), "total_s": None, "test_evaluated": False}

    if evaluate_test:
        model.load_state_dict({k: v.to(device) for k, v in states[sel_round - 1].items()})
        test = data.server["test"]
        rows_t = per_image_metrics(model, DataLoader(test.dataset, batch_size=bs, shuffle=False, **lk), device,
                                   fit_cfg["hd95_normalizer"], fit_cfg["cam_threshold"], with_cam=True)
        with open(os.path.join(out, "test_per_image.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "group_id", "dsc", "hd95_norm", "hd95_px", "ec", "pg"])
            for iid, gid, r in zip(test.image_ids, test.group_ids, rows_t):
                w.writerow([iid, gid, r["dsc"], r["hd95_norm"], r["hd95_px"], r["ec"], r["pg"]])
        meta["test_evaluated"] = True
    meta["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log.info("selected round %d; outputs in %s", sel_round, out)
    return {"out": out, "meta": meta, "rounds": rows, "final_state": states[-1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["breast", "prostate"], required=True)
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, dotted keys")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--no_test", action="store_true", help="skip the final test evaluation (smoke runs)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_experiment(args.task, args.method, args.seed, args.set, args.out, args.tag, evaluate_test=not args.no_test)


if __name__ == "__main__":
    main()
