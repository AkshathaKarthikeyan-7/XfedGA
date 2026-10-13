"""
Plain FedAvg baseline -- no GA. Establishes the pipeline works end to end
and gives a fair reference number to compare FedGA/XFedGA against.

Now tracks and saves the BEST-performing round's model (not just the final
round's), matching the base paper's own evaluation protocol and matching
train_fedga.py's fairness fix -- otherwise FedGA's best-of-N-rounds would be
unfairly compared against FedAvg's last-round-only result.

Usage:
    python experiments/train_baseline_fedavg.py --config configs/breast_clients.yaml
"""
import argparse
import copy
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.breast_dataset import BreastClientDataset, INDEXERS, train_val_split, filter_nonempty_masks
from data.server_split import get_server_split
from models.residual_unet import build_residual_unet
from fedga.fedavg import fedavg_aggregate
from fedga.losses import build_loss
from fedga.ga_optimizer import dice_score


def local_update(model, loader, epochs, lr, device, loss_fn):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, masks)
            loss.backward()
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        print(f"    local epoch {epoch+1}/{epochs} - loss {running_loss/max(n_batches,1):.4f}")
    return model.state_dict()


@torch.no_grad()
def evaluate_dsc(model, loader, device):
    model.eval()
    scores = []
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        out = model(imgs)
        pred = torch.argmax(out, dim=1)
        for p, m in zip(pred, masks):
            scores.append(dice_score(p.float(), m.float()).item())
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breast_clients.yaml")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    client_loaders, client_sizes = [], []
    for client in cfg["clients"]:
        if not os.path.isdir(client["path"]):
            print(f"[skip] {client['name']}: path not found ({client['path']})")
            continue
        indexer = INDEXERS[client["name"]]
        all_pairs = indexer(client["path"])
        print(f"[client] {client['name']}: {len(all_pairs)} raw pairs before filtering")
        all_pairs = filter_nonempty_masks(all_pairs, size=tuple(cfg["image_size"]))
        train_pairs, val_pairs = train_val_split(
            all_pairs, val_ratio=1 - cfg["train_val_split"], seed=cfg["seed"]
        )
        train_ds = BreastClientDataset(client["name"], client["path"], pairs=train_pairs)
        client_loaders.append(DataLoader(train_ds, batch_size=args.batch_size, shuffle=True))
        client_sizes.append(len(train_pairs))
        print(f"[client] {client['name']}: {len(train_pairs)} train pairs")

    if not client_loaders:
        print("No training clients found -- check configs/breast_clients.yaml paths.")
        return

    _, ood_test_pairs = get_server_split(cfg)
    ood_ds = BreastClientDataset(cfg["server"]["name"], cfg["server"]["path"], pairs=ood_test_pairs)
    ood_loader = DataLoader(ood_ds, batch_size=args.batch_size, shuffle=False)
    print(f"[server] {cfg['server']['name']} OOD test set: {len(ood_test_pairs)} pairs")

    global_model = build_residual_unet().to(device)
    loss_fn = build_loss()
    best_ood_dsc = -1.0
    best_round = -1
    round_history = []

    for t in range(args.rounds):
        print(f"\n=== Round {t+1}/{args.rounds} ===")
        client_states = []
        for i, loader in enumerate(client_loaders):
            print(f"  training client {i+1}/{len(client_loaders)}")
            local_model = copy.deepcopy(global_model)
            state = local_update(local_model, loader, args.local_epochs, args.lr, device, loss_fn)
            client_states.append(state)

        aggregated = fedavg_aggregate(client_states, client_sizes)
        global_model.load_state_dict(aggregated)

        ood_dsc = evaluate_dsc(global_model, ood_loader, device)
        print(f"  round {t+1} OOD DSC on {cfg['server']['name']}: {ood_dsc:.4f}")

        if ood_dsc > best_ood_dsc:
            best_ood_dsc = ood_dsc
            best_round = t + 1
            torch.save(global_model.state_dict(), "global_model_fedavg_baseline_BEST.pt")
        round_history.append((t + 1, ood_dsc))

    torch.save(global_model.state_dict(), "global_model_fedavg_baseline.pt")

    import csv
    with open("round_history_fedavg_baseline.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "ood_dsc"])
        writer.writerows(round_history)

    last_k = min(5, len(round_history))
    last_k_scores = [dsc for _, dsc in round_history[-last_k:]]
    import statistics
    mean_last_k = statistics.mean(last_k_scores)
    std_last_k = statistics.pstdev(last_k_scores)

    print("\nSaved: global_model_fedavg_baseline.pt")
    print("Saved round-by-round history: round_history_fedavg_baseline.csv")
    print(f">>> Best single round: DSC = {best_ood_dsc:.4f} at round {best_round} <<<")
    print(f">>> Mean +/- std over last {last_k} rounds (more robust): {mean_last_k:.4f} +/- {std_last_k:.4f} <<<")


if __name__ == "__main__":
    main()