"""
Custom Flower strategy: FedAvg subclass whose aggregate_fit runs the SAME server step as the
custom loop (experiments/run_federated.server_step: q_i-weighted aggregation, GA / random search
on the fixed fitness subsample, validation), plus a message audit.

Message audit: every FitRes must carry exactly len(model.state_dict()) float32 arrays with the
model's shapes, num_examples = n_i, and metrics whose keys are a subset of ALLOWED_METRICS:
  q_i          the client saliency score (only methods that use it)
  client_index the client's identifier (Flower cids are random; the index fixes client order,
               which the noisy-q_i / corrupted-label scenarios and exact parity rely on).
No other key, no images, masks or gradients. Any violation stops the run.
"""
import json
import time

import flwr as fl
import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from experiments.run_federated import server_step
from fl_dist.gpu_lock import release_cached_gpu_memory

ALLOWED_METRICS = {"q_i", "client_index"}


class XFedGAStrategy(fl.server.strategy.FedAvg):
    def __init__(self, cfg, spec, seed, model, srv, device, n_clients, cand_log=None):
        self.cfg, self.spec, self.seed, self.model, self.srv, self.device = cfg, spec, seed, model, srv, device
        self.keys = list(model.state_dict().keys())
        self.shapes = [tuple(v.shape) for v in model.state_dict().values()]
        self.cand_log = cand_log
        self.rows, self.states, self.audit = [], [], {"sent": [], "received": []}
        self._t_fit = None
        init = [v.detach().cpu().numpy() for v in model.state_dict().values()]
        super().__init__(fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=n_clients,
                         min_evaluate_clients=0, min_available_clients=n_clients,
                         initial_parameters=ndarrays_to_parameters(init), on_fit_config_fn=self._fit_config)

    def _fit_config(self, server_round):
        cfg = {"round": server_round}
        self.audit["sent"].append({"round": server_round, "config_keys": sorted(cfg)})
        self._t_fit = time.time()
        return cfg

    def _check(self, server_round, fit_res):
        arrays = parameters_to_ndarrays(fit_res.parameters)
        rec = {"round": server_round, "num_examples": fit_res.num_examples,
               "metric_keys": sorted(fit_res.metrics), "n_arrays": len(arrays),
               "dtypes": sorted({str(a.dtype) for a in arrays}), "bytes": int(sum(a.nbytes for a in arrays)),
               "shapes_ok": [tuple(a.shape) for a in arrays] == self.shapes,
               "client_index": fit_res.metrics.get("client_index")}
        self.audit["received"].append(rec)
        bad = set(fit_res.metrics) - ALLOWED_METRICS
        if bad or not rec["shapes_ok"] or rec["dtypes"] != ["float32"] or len(arrays) != len(self.keys):
            raise RuntimeError(f"message audit failed in round {server_round}: {rec} (unexpected keys {bad})")
        return arrays

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"round {server_round}: {len(failures)} client failures")
        fit_wall = time.time() - self._t_fit
        recs = sorted(((int(r.metrics["client_index"]), self._check(server_round, r), r) for _, r in results),
                      key=lambda x: x[0])
        client_states = [{k: torch.from_numpy(np.array(a, copy=True)).to(self.device) for k, a in zip(self.keys, arr)}
                         for _, arr, _ in recs]
        sizes = [r.num_examples for _, _, r in recs]
        qs = [float(r.metrics["q_i"]) for _, _, r in recs] if self.spec["q_source"] == "gradcam" else []
        t0 = time.time()
        row, best = server_step(self.cfg, self.spec, self.seed, server_round, self.model, client_states, sizes, qs,
                                self.srv, self.device, self.cand_log)
        row.update({"fit_wall_s": round(fit_wall, 2), "server_s": round(time.time() - t0, 2),
                    "bytes_received": int(sum(x["bytes"] for x in self.audit["received"][-len(recs):])),
                    "peak_reserved_MiB": round(torch.cuda.max_memory_reserved() / 2**20) if self.device.type == "cuda" else 0})
        self.rows.append(row)
        self.states.append(best)
        release_cached_gpu_memory()   # memory only: the server holds no cached GPU memory while clients train
        self.sizes = sizes
        return ndarrays_to_parameters([v.numpy() for v in best.values()]), {}

    def audit_summary(self):
        rec = self.audit["received"]
        return {"allowed_metric_keys": sorted(ALLOWED_METRICS),
                "metric_keys_seen": sorted({k for r in rec for k in r["metric_keys"]}),
                "all_shapes_ok": all(r["shapes_ok"] for r in rec), "dtypes_seen": sorted({d for r in rec for d in r["dtypes"]}),
                "config_keys_sent": sorted({k for s in self.audit["sent"] for k in s["config_keys"]}),
                "bytes_per_client_per_round": sorted({r["bytes"] for r in rec}),
                "messages_received": len(rec), "rounds": len(self.audit["sent"]), "detail": self.audit}


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
