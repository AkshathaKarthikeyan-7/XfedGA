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

Outputs in results/runs/<run_id>/: config_resolved.yaml, round_log.csv, ga_candidates.jsonl (fitness
components of every evaluated GA candidate), test_per_image.csv,
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loading import loader_kwargs  # noqa: E402
from data.splits import write_split_log  # noqa: E402
from experiments.config import config_hash, git_commit, load_config, seed_everything, torch_generator  # noqa: E402
from experiments.tasks import ClientData, build_task, split_summary  # noqa: E402
from fedga.client import client_saliency_score, local_update, privatize_update  # noqa: E402
from fedga.evaluation import FixedSet, build_fitness_fn, per_image_metrics, server_client_scores  # noqa: E402
from fedga.fedavg import aggregate, aggregation_weights  # noqa: E402
from fedga.ga_optimizer import ga_evaluation_count, random_search, run_ga, run_mix_ga  # noqa: E402
from data.splits import fitness_halves  # noqa: E402
from fedga.losses import build_loss  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

log = logging.getLogger("run_federated")

# A method is a combination of switches; ablations are rows here, never new code paths.
#   topology:    federated | centralized (pooled client data, upper bound) | local_only (no communication)
#   aggregation: fedavg (n_i/N) | saliency ((n_i/N) * max(q_i, eps)^beta)
#                | server_score ((n_i/N) * max(s_i, eps)^beta; s_i scored by the server on the fitness subsample,
#                  score = dsc_ec (s_i = DSC_i * EC_i) | dsc (s_i = DSC_i); q_i still computed and logged, not used)
#   q_source:    gradcam (client-side Grad-CAM EC) | random (U(0,1) per client per round; ablation F)
#   search:      none | ga | random (GA-free control, same number of fitness evaluations)
#                | mix_ga (GA over client mixing vectors around alpha, same fitness budget as ga)
#   fitness:     dsc | multi (w1*DSC - w2*HD95_norm + w3*EC);  anneal: GA sigma0*exp(-g/tau)
#   prox:        FedProx proximal term (method_options.fedprox_mu)
# Ablation A = fedga, ablation E = xfedga (spec section 5).
_BASE = {"topology": "federated", "aggregation": "fedavg", "q_source": None, "search": "none",
         "fitness": None, "anneal": False, "prox": False, "score": None, "ec": "iou"}


def _m(**kw):
    return {**_BASE, **kw}


METHODS = {
    "fedavg": _m(),
    "fedga": _m(search="ga", fitness="dsc"),
    "xfedga": _m(aggregation="saliency", q_source="gradcam", search="ga", fitness="multi", anneal=True),
    "abl_B": _m(aggregation="saliency", q_source="gradcam"),
    "abl_C": _m(search="ga", fitness="multi"),
    "abl_D": _m(search="ga", fitness="dsc", anneal=True),
    "abl_F": _m(aggregation="saliency", q_source="random", search="ga", fitness="multi", anneal=True),
    "random_search": _m(search="random", fitness="multi"),
    "fedprox": _m(prox=True),
    # XFedGA+ (declared post-v1 variant, plan.md) and its DSC-only control
    # EC for XFedGA+ = share of CAM mass inside the mask (ec="mass"), in s_i and in the fitness EC term
    "xfedga_plus": _m(aggregation="server_score", score="dsc_ec", q_source="gradcam", search="mix_ga",
                      fitness="multi", anneal=True, ec="mass"),
    "xfedga_plus_dsc": _m(aggregation="server_score", score="dsc", q_source="gradcam", search="mix_ga",
                          fitness="multi", anneal=True, ec="mass"),
    "centralized": _m(topology="centralized"),
    "local_only": _m(topology="local_only"),
}


def _mean(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def select_round(val_dsc, window):
    """1-based round with the best centred moving-average validation DSC (ties -> earliest)."""
    half = window // 2
    smooth = [np.mean(val_dsc[max(0, t - half):t + half + 1]) for t in range(len(val_dsc))]
    return int(np.argmax(smooth)) + 1, smooth


class ServerEval:
    """Server-held evaluation data: the fixed fitness subsample (on the device) and the validation set."""

    def __init__(self, cfg, spec, data, device):
        bs = cfg["train"]["batch_size"]
        self.bs = bs
        self.val_loader = DataLoader(data.server["validation"].dataset, batch_size=bs, shuffle=False,
                                     **loader_kwargs(cfg))
        self.val_explain = Subset(data.server["validation"].dataset, data.val_explain_subsample)
        self.fixed = (FixedSet(data.server["fitness"].dataset, data.fitness_subsample, device)
                      if spec["search"] in ("ga", "random", "mix_ga") else None)
        self.fixed_a = self.fixed_b = None
        if cfg["ga"].get("acceptance_check") and self.fixed is not None:
            # acceptance check: the GA (and XFedGA+'s s_i) use half A; half B is used only to accept / reject the winner
            fs = data.server["fitness"]
            groups = [fs.group_ids[i] for i in data.fitness_subsample]
            sizes = [float(self.fixed.masks[k].float().mean()) for k in range(len(self.fixed))]
            pa, pb = fitness_halves(groups, sizes, seed=cfg["seed"])
            self.fixed_a = FixedSet(fs.dataset, [data.fitness_subsample[p] for p in pa], device)
            self.fixed_b = FixedSet(fs.dataset, [data.fitness_subsample[p] for p in pb], device)
            self.halves = {"A": sorted({groups[p] for p in pa}), "B": sorted({groups[p] for p in pb}),
                           "n_images": [len(pa), len(pb)]}


ACCEPTANCE_METHODS = ("xfedga", "xfedga_plus", "xfedga_plus_dsc")


def acceptance_decision(best, w_agg, fitness_b):
    """Keep the GA winner only if its fitness on half B is >= w_agg's; otherwise the round's model is w_agg.
    Returns (state, accepted, fitness_B(winner), fitness_B(w_agg)); 2 fitness evaluations."""
    fb_win, fb_agg = fitness_b(best), fitness_b(w_agg)
    accepted = fb_win >= fb_agg
    return (best if accepted else w_agg), accepted, fb_win, fb_agg


def server_step(cfg, spec, seed, t, model, client_states, sizes, qs, srv, device, cand_log=None):
    """
    Server side of round t, shared by the custom loop and fl_dist/strategy.py:
    q_i handling (random q_i for ablation F, noisy-q_i client) -> aggregation weights alpha_i -> w_agg
    -> search (GA / random search / none) on the fixed fitness subsample -> load the winner into
    `model` -> validation metrics. Returns (round-log row without timing totals, CPU copy of the winner).
    """
    ga_cfg, fit_cfg, sc = cfg["ga"], cfg["fitness"], cfg["scenario"]
    rt = {}
    qs = list(qs)
    if spec["q_source"] == "random":   # ablation F: q_i carries no information about the client
        qs = np.random.RandomState(seed * 1000 + t).uniform(0.0, 1.0, size=len(client_states)).tolist()
    if qs and sc["noisy_qi_client"] is not None:   # robustness: this client always reports q_i = 1
        qs[sc["noisy_qi_client"]] = 1.0
    agg = cfg["aggregation"]
    scores = None
    if spec["aggregation"] == "server_score":   # XFedGA+: server-verified client score on the fitness subsample
        ts = time.time()
        scores = server_client_scores(model, srv.fixed_a or srv.fixed, client_states, spec["score"],
                                      fit_cfg["hd95_normalizer"],
                                      fit_cfg["cam_threshold"], fit_cfg["cam_target"], fit_cfg["cam_layer"],
                                      spec["ec"])
        rt["score_s"] = time.time() - ts
        alpha = aggregation_weights(sizes, [s["s"] for s in scores], agg["beta"], agg["eps"])
    else:
        alpha = aggregation_weights(sizes, qs if spec["aggregation"] == "saliency" else None, agg["beta"], agg["eps"])
    w_agg = aggregate(client_states, alpha)
    client_val = None
    if cfg["evaluation"].get("log_client_val"):   # root-cause check RC4: each client model's validation DSC
        client_val = []
        for st in client_states:
            model.load_state_dict(st)
            client_val.append(_mean(per_image_metrics(model, srv.val_loader, device, fit_cfg["hd95_normalizer"],
                                                      with_cam=False), "dsc"))
    wagg_val = None
    if cfg["evaluation"].get("log_wagg_val"):   # protocol v2: validation DSC of w_agg (GA-overfitting diagnostic)
        model.load_state_dict(w_agg)
        wagg_val = _mean(per_image_metrics(model, srv.val_loader, device, fit_cfg["hd95_normalizer"], with_cam=False),
                         "dsc")

    ts = time.time()
    history, candidates, best_mix = [], [], None
    if spec["search"] in ("ga", "random", "mix_ga"):
        fitness_fn = build_fitness_fn(model, srv.fixed_a or srv.fixed, spec["fitness"], fit_cfg["weights"],
                                      fit_cfg["hd95_normalizer"], fit_cfg["cam_threshold"], record=candidates,
                                      cam_target=fit_cfg["cam_target"], cam_layer=fit_cfg["cam_layer"],
                                      ec_kind=spec["ec"])
    if spec["search"] == "ga":
        best, history = run_ga(w_agg, fitness_fn, mode="xfedga" if spec["anneal"] else "base",
                               population_size=ga_cfg["population"], generations=ga_cfg["generations"],
                               pc=ga_cfg["pc"], pm=ga_cfg["pm"], sigma0=ga_cfg["sigma0"], tau=ga_cfg["tau"],
                               cache_elite_fitness=ga_cfg["cache_elite_fitness"], return_history=True)
    elif spec["search"] == "mix_ga":
        best, best_mix, history, mixes = run_mix_ga(
            client_states, alpha, fitness_fn, population_size=ga_cfg["population"], generations=ga_cfg["generations"],
            pc=ga_cfg["pc"], kappa=ga_cfg["kappa"], tau=ga_cfg["tau"], anneal=spec["anneal"],
            cache_elite_fitness=ga_cfg["cache_elite_fitness"])
        for c, m in zip(candidates, mixes):
            c["mix"] = [round(x, 6) for x in m]
    elif spec["search"] == "random":
        n_eval = ga_evaluation_count(ga_cfg["population"], ga_cfg["generations"], ga_cfg["cache_elite_fitness"])
        best, history = random_search(w_agg, fitness_fn, n_eval, ga_cfg["sigma0"])
    else:
        best = w_agg
    accept = None
    if srv.fixed_b is not None and spec["search"] in ("ga", "mix_ga"):
        fitness_b = build_fitness_fn(model, srv.fixed_b, spec["fitness"], fit_cfg["weights"], fit_cfg["hd95_normalizer"],
                                     fit_cfg["cam_threshold"], cam_target=fit_cfg["cam_target"],
                                     cam_layer=fit_cfg["cam_layer"], ec_kind=spec["ec"])
        best, accepted, fb_win, fb_agg = acceptance_decision(best, w_agg, fitness_b)
        accept = {"accepted": int(accepted), "fit_B_winner": round(fb_win, 6), "fit_B_wagg": round(fb_agg, 6)}
        if not accepted and best_mix is not None:
            best_mix = [round(float(a), 6) for a in alpha]
    rt["ga_s"] = time.time() - ts
    if candidates and cand_log is not None:   # every evaluated candidate, for offline fitness-weight re-ranking
        cand_log.write(json.dumps({"round": t, "candidates": candidates}) + "\n")
        cand_log.flush()
    model.load_state_dict(best)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    ts = time.time()
    val = per_image_metrics(model, srv.val_loader, device, fit_cfg["hd95_normalizer"], with_cam=False)
    expl = per_image_metrics(model, DataLoader(srv.val_explain, batch_size=srv.bs), device, fit_cfg["hd95_normalizer"],
                             fit_cfg["cam_threshold"], with_cam=True, cam_target=fit_cfg["cam_target"],
                             cam_layer=fit_cfg["cam_layer"])
    rt["val_s"] = time.time() - ts
    row = {"round": t, "val_dsc": _mean(val, "dsc"), "val_hd95_norm": _mean(val, "hd95_norm"),
           "val_hd95_px": _mean(val, "hd95_px"), "val_ec": _mean(expl, "ec"), "val_pg": _mean(expl, "pg"),
           "val_ec_mass": _mean(expl, "ec_mass"),
           "q_i": json.dumps([round(q, 6) for q in qs]), "alpha_i": json.dumps([round(a, 6) for a in alpha]),
           "ga_best_per_gen": json.dumps([round(h["best"], 6) for h in history]),
           "ga_mean_per_gen": json.dumps([round(h["mean"], 6) for h in history]),
           "fitness_evals": sum(h["evaluations"] for h in history), **{k: round(v, 2) for k, v in rt.items()}}
    if wagg_val is not None:
        row["val_dsc_wagg"] = wagg_val
    if client_val is not None:
        row["client_val_dsc_i"] = json.dumps([round(x, 6) for x in client_val])
    if scores is not None:
        row["s_i"] = json.dumps([round(s["s"], 6) for s in scores])
        row["client_fit_dsc_i"] = json.dumps([round(s["dsc"], 6) for s in scores])
        row["client_fit_ec_i"] = json.dumps([round(s["ec"], 6) for s in scores])
    if best_mix is not None:
        row["mix_i"] = json.dumps([round(x, 6) for x in best_mix])
    if accept is not None:
        row.update(accept)
    return row, best_state


def _train(cfg, spec, seed, clients, data, device, model, cand_log=None):
    """One training trajectory over `clients` (list of ClientData). Returns (round rows, states, n_i)."""
    tr, ga_cfg, fit_cfg, sc = cfg["train"], cfg["ga"], cfg["fitness"], cfg["scenario"]
    lk = loader_kwargs(cfg)
    bs = tr["batch_size"]
    client_loaders = [DataLoader(c.train, batch_size=bs, shuffle=True, generator=torch_generator(seed * 100 + k), **lk)
                      for k, c in enumerate(clients)]
    qi_loaders = [DataLoader(Subset(c.train, c.qi_indices), batch_size=bs, shuffle=False) for c in clients]
    sizes = [len(c.train) for c in clients]
    srv = ServerEval(cfg, spec, data, device)
    prox_mu = cfg["method_options"]["fedprox_mu"] if spec["prox"] else 0.0
    dp = sc["dp"]
    loss_fn = build_loss()
    rows, states = [], []
    for t in range(1, tr["rounds"] + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        rt = {"local_s": 0.0, "qi_s": 0.0}
        t_round = time.time()
        global_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if dp["enabled"] else None
        client_states, qs, norms = [], [], []
        start_state = {k: v.detach() for k, v in model.state_dict().items()}
        for k, c in enumerate(clients):
            local = copy.deepcopy(model)
            ts = time.time()
            st = local_update(local, client_loaders[k], tr["local_epochs"], tr["lr"], tr["adam_betas"], device,
                              loss_fn, prox_mu=prox_mu)
            if dp["enabled"]:
                st = privatize_update(st, global_state, dp["clip_norm"], dp["noise_multiplier"],
                                      torch_generator(seed * 100000 + t * 100 + k))
                local.load_state_dict(st)
            client_states.append(st)
            # L2 norm of the update w_i - w_global (sent update, after DP if enabled): calibrates dp.clip_norm
            norms.append(float(torch.sqrt(sum(((st[n].float() - start_state[n].float()) ** 2).sum()
                                              for n in st if st[n].dtype.is_floating_point))))
            rt["local_s"] += time.time() - ts
            if spec["q_source"] == "gradcam":
                ts = time.time()
                qs.append(client_saliency_score(local, qi_loaders[k], device, fit_cfg["cam_threshold"],
                                                fit_cfg["cam_target"], fit_cfg["cam_layer"]))
                rt["qi_s"] += time.time() - ts
        row, best_state = server_step(cfg, spec, seed, t, model, client_states, sizes, qs, srv, device, cand_log)
        states.append(best_state)
        row["update_norm_i"] = json.dumps([round(x, 4) for x in norms])
        timing = {"local_s": round(rt["local_s"], 2), "qi_s": round(rt["qi_s"], 2),
                  "ga_s": row.pop("ga_s"), "val_s": row.pop("val_s")}
        row.update(timing)   # same column order as before the server_step refactor
        row["round_s"] = round(time.time() - t_round, 2)
        row["peak_reserved_MiB"] = round(torch.cuda.max_memory_reserved() / 2**20) if device.type == "cuda" else 0
        rows.append(row)
        log.info("round %d/%d val DSC %.4f HD95n %.4f EC %.4f PG %.3f q=%s (%.0fs)", t, tr["rounds"],
                 row["val_dsc"], row["val_hd95_norm"], row["val_ec"], row["val_pg"], row["q_i"], row["round_s"])
    return rows, states, sizes


def _finish(cfg, data, device, model, rows, states, out, suffix, evaluate_test):
    """Write the round log, choose the validation-selected round, evaluate the test set once."""
    fit_cfg, bs = cfg["fitness"], cfg["train"]["batch_size"]
    with open(os.path.join(out, f"round_log{suffix}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sel_round, smooth = select_round([r["val_dsc"] for r in rows], cfg["evaluation"]["selection_smoothing"])
    tested = False
    if evaluate_test:
        model.load_state_dict({k: v.to(device) for k, v in states[sel_round - 1].items()})
        test = data.server["test"]
        rows_t = per_image_metrics(model, DataLoader(test.dataset, batch_size=bs, shuffle=False, **loader_kwargs(cfg)),
                                   device, fit_cfg["hd95_normalizer"], fit_cfg["cam_threshold"], with_cam=True,
                                   cam_target=fit_cfg["cam_target"], cam_layer=fit_cfg["cam_layer"])
        with open(os.path.join(out, f"test_per_image{suffix}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "group_id", "dsc", "hd95_norm", "hd95_px", "ec", "pg"])
            for iid, gid, r in zip(test.image_ids, test.group_ids, rows_t):
                w.writerow([iid, gid, r["dsc"], r["hd95_norm"], r["hd95_px"], r["ec"], r["pg"]])
        tested = True
    return sel_round, smooth, tested


def run_experiment(task, method, seed, overrides=None, out_root="results/runs", tag="", evaluate_test=True):
    """Run one (task, method, seed) experiment end to end and write its outputs to `out_root`."""
    seed_everything(seed)
    cfg = load_config(task, overrides)
    spec = METHODS[method]
    if cfg["ga"].get("acceptance_check") and method not in ACCEPTANCE_METHODS:
        raise ValueError(f"ga.acceptance_check is pre-registered for {ACCEPTANCE_METHODS} only, not {method}")
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
    part = cfg["scenario"]["partition"]
    name = f"{task}_dataseed{cfg['seed']}" + ("" if part == "random" else f"_{part}") + ".json"
    shared = os.path.join("results", "splits", name)
    summary = split_summary(data)
    if os.path.exists(shared):
        with open(shared, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(summary)):
                raise RuntimeError(f"splits differ from {shared}: data or split code changed")
    else:
        write_split_log(shared, summary)
    setup_s = time.time() - t0

    meta = {"run_id": run_id, "task": task, "method": method, "method_switches": spec, "seed": seed,
            "config_hash": chash, "git_commit": git_commit(), "selection": "centred moving-average validation DSC",
            "clients": [c.name for c in data.clients],
            "server_sizes": {r: len(s.image_ids) for r, s in data.server.items()},
            "setup_s": round(setup_s, 1), "total_s": None}
    model = build_residual_unet().to(device)
    if spec["topology"] == "local_only":   # lower bound: each client trains alone, no communication
        init = copy.deepcopy(model.state_dict())
        meta["per_client"] = {}
        for k, c in enumerate(data.clients):
            model.load_state_dict(init)
            rows, states, sizes = _train(cfg, spec, seed, [c], data, device, model)
            sel, smooth, tested = _finish(cfg, data, device, model, rows, states, out, f"_client{k + 1}", evaluate_test)
            torch.save(states[sel - 1], os.path.join(out, f"{task}_{method}_client{k + 1}_seed{seed}_valselected.pt"))
            meta["per_client"][c.name] = {"selected_round": sel, "smoothed_val_dsc": smooth, "n_i": sizes,
                                          "test_evaluated": tested}
        meta["test_evaluated"] = evaluate_test
    else:
        clients = data.clients
        if spec["topology"] == "centralized":   # upper bound: one model on the pooled client data
            pooled = ConcatDataset([c.train for c in data.clients])
            offsets = np.cumsum([0] + [len(c.train) for c in data.clients[:-1]])
            qi = [int(o) + i for o, c in zip(offsets, data.clients) for i in c.qi_indices]
            clients = [ClientData("pooled", pooled, qi, sum(c.n_train_groups for c in data.clients))]
        with open(os.path.join(out, "ga_candidates.jsonl"), "w", encoding="utf-8") as cand_log:
            rows, states, sizes = _train(cfg, spec, seed, clients, data, device, model, cand_log)
        sel, smooth, tested = _finish(cfg, data, device, model, rows, states, out, "", evaluate_test)
        torch.save(states[sel - 1], os.path.join(out, f"{task}_{method}_seed{seed}_valselected.pt"))
        meta.update({"selected_round": sel, "smoothed_val_dsc": smooth, "n_i": sizes, "test_evaluated": tested})
    meta["total_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log.info("outputs in %s", out)
    return {"out": out, "meta": meta, "rounds": rows, "final_state": states[-1]}


def main():
    """Command-line entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["breast", "prostate", "prostate_ms", "prostate_ms_runmc"], required=True)
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
