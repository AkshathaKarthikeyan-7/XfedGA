"""
Flower server process: holds the server fitness / validation / test sets, runs aggregation + GA +
validation through XFedGAStrategy, then selects the checkpoint on validation and evaluates the test
set once, with the same code and outputs as the custom loop (plus message_audit.json).

Normally started by fl_dist/launch.py. Outputs go to --out_dir.
"""
import argparse
import json
import logging
import os
import sys
import time

import flwr as fl
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import config_hash, git_commit, load_config, seed_everything  # noqa: E402
from experiments.run_federated import METHODS, ServerEval, _finish  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from fl_dist.strategy import XFedGAStrategy, dump  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

log = logging.getLogger("fl_dist.server")


def main():
    """Command-line entry point for the Flower server process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no_test", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    seed_everything(args.seed)                      # same order as the custom loop: seed -> data -> model
    cfg = load_config(args.task, args.set)
    spec = METHODS[args.method]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config_resolved.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"task": args.task, "method": args.method, "method_switches": spec, "seed": args.seed,
                        "framework": "fl_dist (Flower 1.7.0, multi-process, loopback gRPC)", **cfg}, f, sort_keys=False)
    data = build_task(args.task, cfg, part="server")
    model = build_residual_unet().to(device)
    srv = ServerEval(cfg, spec, data, device)
    n_clients = len(cfg["clients"])
    with open(os.path.join(args.out_dir, "ga_candidates.jsonl"), "w", encoding="utf-8") as cand_log:
        strategy = XFedGAStrategy(cfg, spec, args.seed, model, srv, device, n_clients, cand_log)
        fl.server.start_server(server_address=args.server_address,
                               config=fl.server.ServerConfig(num_rounds=cfg["train"]["rounds"]), strategy=strategy)
    if len(strategy.rows) != cfg["train"]["rounds"]:
        raise RuntimeError(f"only {len(strategy.rows)} of {cfg['train']['rounds']} rounds completed")
    sel, smooth, tested = _finish(cfg, data, device, model, strategy.rows, strategy.states, args.out_dir, "",
                                  evaluate_test=not args.no_test)
    torch.save(strategy.states[sel - 1],
               os.path.join(args.out_dir, f"{args.task}_{args.method}_seed{args.seed}_fl_valselected.pt"))
    dump(os.path.join(args.out_dir, "message_audit.json"), strategy.audit_summary())
    meta = {"task": args.task, "method": args.method, "seed": args.seed, "framework": "fl_dist",
            "config_hash": config_hash({"cfg": cfg, "method": args.method}), "git_commit": git_commit(),
            "selected_round": sel, "smoothed_val_dsc": smooth, "n_i": strategy.sizes, "test_evaluated": tested,
            "server_sizes": {r: len(s.image_ids) for r, s in data.server.items()},
            "server_peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if device.type == "cuda" else 0,
            "total_s": round(time.time() - t0, 1)}
    with open(os.path.join(args.out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    log.info("server finished: selected round %d, outputs in %s", sel, args.out_dir)


if __name__ == "__main__":
    main()
