"""
Federated training on the prostate segmentation task (PROMISE12 + Decathlon
Task05_Prostate), reusing FedAvg/FedGA/XFedGA logic from train_fedga.py.

Two clients: PROMISE12 split randomly 50/50 by case (no site metadata was
available for this download, so this is a SYNTHETIC non-IID partition,
unlike the breast task's real-institution 2-dataset split).
Held-out OOD set: Decathlon Task05_Prostate (T2 channel, PZ+TZ merged to
binary), split at the case level into a fitness-eval half (used during GA
candidate scoring) and a true OOD-test half, exactly mirroring the breast
task's BUS-BRA fitness/OOD split.

Usage (small correctness check first):
    python experiments/train_fedga_prostate.py --mode xfedga --rounds 1 --population 4 --generations 2

Usage (full run):
    python experiments/train_fedga_prostate.py --mode xfedga --rounds 15 --population 6 --generations 5
"""
import argparse
import copy
import os
import sys
import time
import csv
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.prostate_dataset import ProstateClientDataset, split_promise12_clients
from data.server_split import get_prostate_server_split
from data.loading import loader_kwargs
from models.residual_unet import build_residual_unet
from fedga.fedavg import fedavg_aggregate, explanation_weighted_aggregate
from fedga.losses import build_loss

from experiments.train_fedga import (
    local_update,
    evaluate_dsc,
    evaluate_explainability,
    build_fitness_evaluator,
    compute_client_saliency_score,
)
from fedga.ga_optimizer import run_ga


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prostate_clients.yaml")
    parser.add_argument("--mode", choices=["base", "xfedga"], default="base")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--pc", type=float, default=0.5)
    parser.add_argument("--pm", type=float, default=0.1)
    parser.add_argument("--sigma0", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.5,
                         help="Saliency-weighting exponent in explanation_weighted_aggregate "
                              "(higher = more saliency influence; lower = more dampened, "
                              "closer to plain size-weighted FedAvg)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | mode: {args.mode} | population: {args.population} | generations: {args.generations}")

    promise12_root = cfg["clients"][0]["path"]
    client1_cases, client2_cases = split_promise12_clients(promise12_root, seed=cfg.get("seed", 42))

    client1_ds = ProstateClientDataset("PROMISE12", promise12_root, cases=client1_cases)
    client2_ds = ProstateClientDataset("PROMISE12", promise12_root, cases=client2_cases)
    client_loaders = [
        DataLoader(client1_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs(cfg)),
        DataLoader(client2_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs(cfg)),
    ]
    client_sizes = [len(client1_ds), len(client2_ds)]
    print(f"[client] PROMISE12 client_1: {len(client1_ds)} slices")
    print(f"[client] PROMISE12 client_2: {len(client2_ds)} slices")

    fitness_cases, ood_cases = get_prostate_server_split(cfg)
    decathlon_root = cfg["server"]["path"]
    fitness_ds = ProstateClientDataset("Decathlon", decathlon_root, cases=fitness_cases)
    ood_ds = ProstateClientDataset("Decathlon", decathlon_root, cases=ood_cases)
    fitness_loader = DataLoader(fitness_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs(cfg))
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs(cfg))
    print(f"[server] Decathlon: {len(fitness_ds)} fitness-eval slices, {len(ood_ds)} OOD-test slices")

    global_model = build_residual_unet().to(device)
    loss_fn = build_loss()
    best_ood_dsc = -1.0
    best_state = None
    best_round = -1
    round_history = []

    for t in range(args.rounds):
        round_start = time.time()
        print(f"\n=== Round {t+1}/{args.rounds} ===")

        client_states = []
        client_saliency_scores = []
        for i, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            state = local_update(local_model, loader, args.local_epochs, args.lr, device, loss_fn)
            client_states.append(state)
            print(f"  client {i+1}/{len(client_loaders)} local training done")

            if args.mode == "xfedga":
                local_model.load_state_dict(state)
                score = compute_client_saliency_score(local_model, loader, device)
                client_saliency_scores.append(score)
                print(f"    client {i+1} saliency-quality score: {score:.4f}")

        if args.mode == "xfedga":
            aggregated = explanation_weighted_aggregate(client_states, client_sizes, client_saliency_scores, beta=args.beta)
        else:
            aggregated = fedavg_aggregate(client_states, client_sizes)

        evaluate_fn = build_fitness_evaluator(global_model, fitness_loader, device, args.mode)
        best_candidate_state = run_ga(
            aggregated, evaluate_fn, mode=args.mode,
            population_size=args.population, generations=args.generations,
            pc=args.pc, pm=args.pm, sigma0=args.sigma0,
        )
        global_model.load_state_dict(best_candidate_state)

        ood_dsc = evaluate_dsc(global_model, ood_loader, device)
        ood_ec, ood_pg = evaluate_explainability(global_model, ood_loader, device)
        elapsed = time.time() - round_start
        print(f"  round {t+1} OOD DSC on Decathlon: {ood_dsc:.4f}  |  EC: {ood_ec:.4f}  |  pointing-game: {ood_pg:.4f}  ({elapsed:.1f}s)")

        if ood_dsc > best_ood_dsc:
            best_ood_dsc = ood_dsc
            best_state = copy.deepcopy(global_model.state_dict())
            best_round = t + 1
        saliency_str = ",".join(f"{s:.4f}" for s in client_saliency_scores) if client_saliency_scores else ""
        round_history.append((t + 1, ood_dsc, ood_ec, ood_pg, saliency_str))

    out_name = f"global_model_{args.mode}_fedga_prostate.pt"
    torch.save(global_model.state_dict(), out_name)
    best_out_name = f"global_model_{args.mode}_fedga_prostate_BEST.pt"
    torch.save(best_state, best_out_name)

    csv_name = f"round_history_{args.mode}_fedga_prostate.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "ood_dsc", "ood_ec", "ood_pointing_game", "client_saliency_scores"])
        writer.writerows(round_history)

    last_k = min(5, len(round_history))
    last_k_rows = round_history[-last_k:]
    mean_last_k = float(np.mean([r[1] for r in last_k_rows]))
    std_last_k = float(np.std([r[1] for r in last_k_rows]))
    mean_ec_last_k = float(np.mean([r[2] for r in last_k_rows]))
    mean_pg_last_k = float(np.mean([r[3] for r in last_k_rows]))

    print(f"\nSaved final-round model: {out_name}")
    print(f"Saved BEST model (round {best_round}, OOD DSC {best_ood_dsc:.4f}): {best_out_name}")
    print(f"Saved round-by-round history: {csv_name}")
    print(f"\n>>> Best single round: DSC = {best_ood_dsc:.4f} at round {best_round} <<<")
    print(f">>> Mean +/- std over last {last_k} rounds (more robust): {mean_last_k:.4f} +/- {std_last_k:.4f} <<<")
    print(f">>> Mean explanation-consistency (last {last_k} rounds): {mean_ec_last_k:.4f} <<<")
    print(f">>> Mean pointing-game accuracy (last {last_k} rounds): {mean_pg_last_k:.4f} <<<")


if __name__ == "__main__":
    main()




