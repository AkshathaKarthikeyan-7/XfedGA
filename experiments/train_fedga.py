"""
Federated training WITH the GA server-side optimization step.

Two modes:
    --mode base    : reproduces base FedGA exactly (DSC-only fitness, fixed
                      mutation, plain FedAvg aggregation)
    --mode xfedga   : your contribution --
                      - multi-objective fitness (DSC + HD95 + explanation
                        consistency) driving the GA's candidate selection
                      - annealed mutation (sigma decays per generation)
                      - explanation-weighted client aggregation (each
                        client's saliency-mask alignment, on its own data,
                        scales its influence on the aggregated model)

Usage (small correctness check first):
    python experiments/train_fedga.py --mode xfedga --rounds 1 --population 4 --generations 2

Usage (full run):
    python experiments/train_fedga.py --mode xfedga --rounds 15 --population 6 --generations 5
"""
import argparse
import copy
import os
import sys
import time
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.breast_dataset import BreastClientDataset, INDEXERS, train_val_split, filter_nonempty_masks
from data.server_split import get_server_split
from models.residual_unet import build_residual_unet
from fedga.fedavg import fedavg_aggregate, explanation_weighted_aggregate
from fedga.losses import build_loss
from fedga.ga_optimizer import run_ga, dice_score, hd95, multi_objective_fitness
from xai.explainability import get_gradcam, explanation_consistency


def _gradcam_target_layer(model):
    # last decoder block before final output -- 16 channels, 192x192 (see inspect_model.py)
    return model.model[1].submodule[2][1].conv.unit0.conv


def local_update(model, loader, epochs, lr, device, loss_fn):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    for _ in range(epochs):
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            opt.step()
    return model.state_dict()


@torch.no_grad()
def evaluate_dsc(model, loader, device):
    model.eval()
    scores = []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        pred = torch.argmax(model(imgs), dim=1)
        for p, m in zip(pred, masks):
            scores.append(dice_score(p.float(), m.float()).item())
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_explainability(model, loader, device, max_samples=30):
    """
    Mean explanation-consistency and pointing-game accuracy over a capped
    sample of the OOD test set -- run once per round (not per GA candidate,
    unlike evaluate_multi_objective) since this is for REPORTING the
    accuracy-explainability trade-off, not for driving the search itself.
    """
    from xai.explainability import pointing_game_accuracy
    model.eval()
    target_layer = _gradcam_target_layer(model)
    ecs, pgs = [], []
    seen = 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            if seen >= max_samples:
                break
            img = imgs[i:i+1].to(device)
            mask = masks[i].numpy()
            try:
                cam = get_gradcam(model, target_layer, img, target_class=1)
                ecs.append(explanation_consistency(cam, mask, threshold=0.5))
                pgs.append(pointing_game_accuracy(cam, mask))
            except Exception:
                pass
            seen += 1
        if seen >= max_samples:
            break
    mean_ec = float(np.mean(ecs)) if ecs else 0.0
    pg_accuracy = float(np.mean(pgs)) if pgs else 0.0
    return mean_ec, pg_accuracy


def evaluate_multi_objective(model, loader, device, max_samples=40):
    """
    DSC + HD95 + explanation-consistency, averaged over a capped subsample
    (Grad-CAM per image is expensive and this runs per GA candidate per
    generation, every round -- max_samples=40 balances signal vs. compute).
    """
    model.eval()
    target_layer = _gradcam_target_layer(model)
    dscs, hd95s, ecs = [], [], []
    seen = 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            if seen >= max_samples:
                break
            img = imgs[i:i+1].to(device)
            mask = masks[i].numpy()

            with torch.no_grad():
                pred = torch.argmax(model(img), dim=1)[0]
            d = dice_score(pred.float(), torch.from_numpy(mask).float().to(device)).item()
            h = hd95(pred.float(), torch.from_numpy(mask).float().to(device))

            try:
                cam = get_gradcam(model, target_layer, img, target_class=1)
                ec = explanation_consistency(cam, mask, threshold=0.5)
            except Exception:
                ec = 0.0

            dscs.append(d); hd95s.append(h); ecs.append(ec)
            seen += 1
        if seen >= max_samples:
            break

    mean_dsc = float(np.mean(dscs)) if dscs else 0.0
    mean_hd95 = float(np.mean(hd95s)) if hd95s else 1.0
    mean_ec = float(np.mean(ecs)) if ecs else 0.0
    return multi_objective_fitness(mean_dsc, mean_hd95, mean_ec)


def compute_client_saliency_score(model, loader, device, max_samples=40):
    """Per-client explanation quality on its own data, feeds aggregation weight."""
    model.eval()
    target_layer = _gradcam_target_layer(model)
    ecs = []
    seen = 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            if seen >= max_samples:
                break
            img = imgs[i:i+1].to(device)
            mask = masks[i].numpy()
            try:
                cam = get_gradcam(model, target_layer, img, target_class=1)
                ecs.append(explanation_consistency(cam, mask, threshold=0.5))
            except Exception:
                ecs.append(0.0)
            seen += 1
        if seen >= max_samples:
            break
    return float(np.mean(ecs)) if ecs else 0.5


def build_fitness_evaluator(model_template, fitness_loader, device, mode):
    def evaluate_fn(state_dict):
        model = copy.deepcopy(model_template)
        model.load_state_dict(state_dict)
        model.to(device)
        if mode == "xfedga":
            return evaluate_multi_objective(model, fitness_loader, device)
        return evaluate_dsc(model, fitness_loader, device)
    return evaluate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breast_clients.yaml")
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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | mode: {args.mode} | population: {args.population} | generations: {args.generations}")

    client_loaders, client_sizes = [], []
    for client in cfg["clients"]:
        if not os.path.isdir(client["path"]):
            print(f"[skip] {client['name']}: path not found ({client['path']})")
            continue
        indexer = INDEXERS[client["name"]]
        all_pairs = indexer(client["path"])
        all_pairs = filter_nonempty_masks(all_pairs, size=tuple(cfg["image_size"]))
        train_pairs, _ = train_val_split(all_pairs, val_ratio=1 - cfg["train_val_split"], seed=cfg["seed"])
        train_ds = BreastClientDataset(client["name"], client["path"], pairs=train_pairs)
        client_loaders.append(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True))
        client_sizes.append(len(train_pairs))
        print(f"[client] {client['name']}: {len(train_pairs)} filtered train pairs")

    if not client_loaders:
        print("No training clients found -- check configs/breast_clients.yaml paths.")
        return

    fitness_pairs, ood_test_pairs = get_server_split(cfg)
    fitness_ds = BreastClientDataset(cfg["server"]["name"], cfg["server"]["path"], pairs=fitness_pairs)
    fitness_loader = DataLoader(fitness_ds, batch_size=args.batch_size, shuffle=False)
    ood_ds = BreastClientDataset(cfg["server"]["name"], cfg["server"]["path"], pairs=ood_test_pairs)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[server] {cfg['server']['name']}: {len(fitness_pairs)} fitness-eval pairs, {len(ood_test_pairs)} OOD-test pairs")

    global_model = build_residual_unet().to(device)
    loss_fn = build_loss()
    best_ood_dsc = -1.0
    best_state = None
    best_round = -1
    round_history = []  # (round, ood_dsc) -- for mean/std reporting, not just best-round

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
            aggregated = explanation_weighted_aggregate(client_states, client_sizes, client_saliency_scores)
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
        print(f"  round {t+1} OOD DSC on {cfg['server']['name']}: {ood_dsc:.4f}  |  EC: {ood_ec:.4f}  |  pointing-game: {ood_pg:.4f}  ({elapsed:.1f}s)")

        if ood_dsc > best_ood_dsc:
            best_ood_dsc = ood_dsc
            best_state = copy.deepcopy(global_model.state_dict())
            best_round = t + 1
        round_history.append((t + 1, ood_dsc, ood_ec, ood_pg))

    out_name = f"global_model_{args.mode}_fedga.pt"
    torch.save(global_model.state_dict(), out_name)
    best_out_name = f"global_model_{args.mode}_fedga_BEST.pt"
    torch.save(best_state, best_out_name)

    import csv
    csv_name = f"round_history_{args.mode}_fedga.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "ood_dsc", "ood_ec", "ood_pointing_game"])
        writer.writerows(round_history)

    last_k = min(5, len(round_history))
    last_k_rows = round_history[-last_k:]
    last_k_scores = [r[1] for r in last_k_rows]
    last_k_ec = [r[2] for r in last_k_rows]
    last_k_pg = [r[3] for r in last_k_rows]
    mean_last_k = float(np.mean(last_k_scores))
    std_last_k = float(np.std(last_k_scores))
    mean_ec_last_k = float(np.mean(last_k_ec))
    mean_pg_last_k = float(np.mean(last_k_pg))

    print(f"\nSaved final-round model: {out_name}")
    print(f"Saved BEST model (round {best_round}, OOD DSC {best_ood_dsc:.4f}): {best_out_name}")
    print(f"Saved round-by-round history: {csv_name}")
    print(f"\n>>> Best single round: DSC = {best_ood_dsc:.4f} at round {best_round} <<<")
    print(f">>> Mean +/- std over last {last_k} rounds (more robust): {mean_last_k:.4f} +/- {std_last_k:.4f} <<<")
    print(f">>> Mean explanation-consistency (last {last_k} rounds): {mean_ec_last_k:.4f} <<<")
    print(f">>> Mean pointing-game accuracy (last {last_k} rounds): {mean_pg_last_k:.4f} <<<")


if __name__ == "__main__":
    main()
