"""
Carves the server-designated hospital's data into two disjoint halves:
  - fitness_pairs : used for GA fitness evaluation (not needed yet for the
                     plain FedAvg baseline, but prepared now so the later
                     GA step is a drop-in addition, not a rewrite)
  - ood_test_pairs : true held-out out-of-distribution test set, used to
                     report final DSC/HD95 the same way the base paper's
                     Table V does

Run directly to sanity-check the split sizes:
    python data/server_split.py --config configs/breast_clients.yaml
"""
import argparse
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.breast_dataset import INDEXERS, train_val_split, filter_nonempty_masks


def get_server_split(cfg):
    server_cfg = cfg["server"]
    indexer = INDEXERS[server_cfg["name"]]
    all_pairs = indexer(server_cfg["path"])
    all_pairs = filter_nonempty_masks(all_pairs, size=tuple(cfg.get("image_size", (384, 384))))

    # reuse train_val_split as a generic 50/50 splitter
    fitness_pairs, ood_test_pairs = train_val_split(
        all_pairs, val_ratio=server_cfg.get("fitness_split", 0.5), seed=server_cfg.get("seed", 42)
    )
    return fitness_pairs, ood_test_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/breast_clients.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    fitness_pairs, ood_test_pairs = get_server_split(cfg)
    print(f"Server hospital: {cfg['server']['name']}")
    print(f"  fitness-eval pairs : {len(fitness_pairs)}")
    print(f"  OOD test pairs     : {len(ood_test_pairs)}")