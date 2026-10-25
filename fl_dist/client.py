"""
Flower client: one OS process per client. Loads ONLY its own client's data, trains locally and
computes q_i on its own fixed q_i subsample, and returns (weights w_i, n_i,
{"q_i": q_i, "client_index": k}); the index is an identifier, see fl_dist/strategy.py.

    & $py fl_dist/client.py --task breast --method xfedga --seed 1 --client 0 --server_address 127.0.0.1:8080
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loading import loader_kwargs  # noqa: E402
from experiments.config import load_config, seed_everything, torch_generator  # noqa: E402
from experiments.run_federated import METHODS  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from fedga.client import client_saliency_score, local_update, privatize_update  # noqa: E402
from fedga.losses import build_loss  # noqa: E402
from fl_dist.gpu_lock import gpu_lock, release_cached_gpu_memory  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402

log = logging.getLogger("fl_dist.client")


def state_to_ndarrays(state):
    """Model state dict -> list of numpy arrays (the Flower message format)."""
    return [v.detach().cpu().numpy() for v in state.values()]


def ndarrays_to_state(keys, arrays, device):
    """List of numpy arrays -> state dict on `device`, in the order of `keys`."""
    return {k: torch.from_numpy(np.array(a, copy=True)).to(device) for k, a in zip(keys, arrays)}


class XFedGAClient(fl.client.NumPyClient):
    """Flower client: trains locally on its own data and returns (w_i, n_i, {q_i, client index})."""
    def __init__(self, task, method, seed, client, overrides, device, lock_path=None):
        self.cfg = load_config(task, overrides)
        self.spec = METHODS[method]
        if self.spec["topology"] != "federated":
            raise ValueError(f"{method} is not a federated method")
        self.seed, self.k, self.device = seed, client, device
        data = build_task(task, self.cfg, part=client)          # this client's data only
        self.data = data.clients[0]
        tr = self.cfg["train"]
        # same loader seeding as the custom loop (experiments/run_federated._train)
        self.loader = DataLoader(self.data.train, batch_size=tr["batch_size"], shuffle=True,
                                 generator=torch_generator(seed * 100 + client), **loader_kwargs(self.cfg))
        self.qi_loader = DataLoader(Subset(self.data.train, self.data.qi_indices), batch_size=tr["batch_size"],
                                    shuffle=False)
        self.model = build_residual_unet().to(device)
        self.keys = list(self.model.state_dict().keys())
        self.loss_fn = build_loss()
        self.received = []                                       # message audit, client side
        self.lock_path, self.lock_log = lock_path, []           # cross-process GPU lock (fl_dist/gpu_lock.py)
        log.info("client %d (%s): %d training samples", client, self.data.name, len(self.data.train))

    def get_parameters(self, config):
        """Current local model weights."""
        return state_to_ndarrays(self.model.state_dict())

    def fit(self, parameters, config):
        """One round: load the global model, train locally, compute q_i; returns weights, n_i and metrics."""
        self.received.append({"round": config.get("round"), "config_keys": sorted(config.keys()),
                              "n_arrays": len(parameters)})
        t = int(config["round"])
        with gpu_lock(self.lock_path) as waited:   # one client on the GPU at a time; results unchanged, timing only
            t_held = time.time()
            out = self._fit(parameters, t)
            release_cached_gpu_memory()           # idle clients must not hold cached GPU memory
        self.lock_log.append({"round": t, "waited_s": round(waited, 2), "held_s": round(time.time() - t_held, 2)})
        log.info("round %d: GPU lock waited %.1f s, held %.1f s", t, waited, time.time() - t_held)
        return out

    def _fit(self, parameters, t):
        tr, sc = self.cfg["train"], self.cfg["scenario"]
        global_state = ndarrays_to_state(self.keys, parameters, self.device)
        self.model.load_state_dict(global_state)
        local = copy.deepcopy(self.model)
        prox_mu = self.cfg["method_options"]["fedprox_mu"] if self.spec["prox"] else 0.0
        state = local_update(local, self.loader, tr["local_epochs"], tr["lr"], tr["adam_betas"], self.device,
                             self.loss_fn, prox_mu=prox_mu)
        if sc["dp"]["enabled"]:
            state = privatize_update(state, global_state, sc["dp"]["clip_norm"], sc["dp"]["noise_multiplier"],
                                     torch_generator(self.seed * 100000 + t * 100 + self.k))
            local.load_state_dict(state)
        metrics = {"client_index": self.k}   # identifier only (see fl_dist/strategy.py, message audit)
        if self.spec["q_source"] == "gradcam":
            metrics["q_i"] = float(client_saliency_score(local, self.qi_loader, self.device,
                                                         self.cfg["fitness"]["cam_threshold"],
                                                         self.cfg["fitness"]["cam_target"], self.cfg["fitness"]["cam_layer"]))
        log.info("round %d: trained; metrics %s", t, metrics)
        return state_to_ndarrays(state), len(self.data.train), metrics

    def evaluate(self, parameters, config):   # evaluation is server-side only (fraction_evaluate = 0)
        """Not used: evaluation is server-side only."""
        return 0.0, 0, {}


def main():
    """Command-line entry point for one client process."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--client", type=int, required=True)
    ap.add_argument("--server_address", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--audit_out", default=None)
    ap.add_argument("--gpu_lock", default=None, help="lock file shared by the clients of one run (launch.py sets it)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client = XFedGAClient(args.task, args.method, args.seed, args.client, args.set, device, args.gpu_lock)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    if args.audit_out:
        with open(args.audit_out, "w", encoding="utf-8") as f:
            json.dump({"client": args.client, "received": client.received, "gpu_lock": client.lock_log,
                       "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if device.type == "cuda" else 0},
                      f, indent=1)
    log.info("client %d finished", args.client)


if __name__ == "__main__":
    main()
