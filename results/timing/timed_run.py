"""
Session-1 timing harness. Runs the EXISTING training scripts unmodified, with
timing wrappers monkey-patched onto their module-level functions at runtime.

Why a wrapper: the audit session may not edit existing files, and the
training scripts write *.pt / *.csv into the current working directory
(which would overwrite the root-level checkpoints). This harness:
  - writes a copy of the task config with absolute data paths into --workdir
  - chdirs into --workdir, so all script outputs land there
  - records per-round: wall clock, local training, q_i, aggregation, GA
    (with fitness-eval count and time), OOD evaluation, peak GPU memory
  - stops after 3 rounds if the projected 15-round wall clock > --cap_hours

Usage (from repo root):
    python results/timing/timed_run.py --task breast --mode base --workdir results/timing/runs/breast_base -- --rounds 15
Everything after `--` is passed to the training script verbatim.
"""
import argparse
import json
import os
import sys
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

import torch  # noqa: E402

CONFIGS = {"breast": "configs/breast_clients.yaml", "prostate": "configs/prostate_clients.yaml"}
MODULES = {"breast": "experiments.train_fedga", "prostate": "experiments.train_fedga_prostate"}


class StopEarly(Exception):
    pass


class Recorder:
    def __init__(self, path, cap_hours, total_rounds):
        self.f = open(path, "w")
        self.depth = 0
        self.cur = None
        self.round = 0
        self.t_start = time.time()
        self.t_round_start = None
        self.round_times = []
        self.cap_s = cap_hours * 3600
        self.total_rounds = total_rounds
        self.new_round()

    def new_round(self):
        self.round += 1
        self.cur = {"round": self.round, "local_train_s": 0.0, "qi_s": 0.0, "agg_s": 0.0,
                    "ga_s": 0.0, "fitness_eval_s": 0.0, "n_fitness_evals": 0,
                    "ood_eval_s": 0.0, "n_local_updates": 0}

    def log(self, obj):
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def wrap(self, fn, key, count_key=None, on_end=None):
        rec = self

        def wrapped(*a, **kw):
            outer = rec.depth == 0
            if outer and rec.t_round_start is None:
                rec.t_round_start = time.time()
                rec.log({"event": "setup_done", "setup_s": rec.t_round_start - rec.t_start})
            rec.depth += 1
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.time() - t0
                rec.depth -= 1
                if outer:
                    rec.cur[key] += dt
                    if count_key:
                        rec.cur[count_key] += 1
                    if on_end:
                        on_end()
        return wrapped

    def wrap_fitness(self, fn):
        rec = self

        def wrapped(*a, **kw):
            t0 = time.time()
            try:
                return fn(*a, **kw)
            finally:
                rec.cur["fitness_eval_s"] += time.time() - t0
                rec.cur["n_fitness_evals"] += 1
        return wrapped

    def end_round(self):
        now = time.time()
        c = self.cur
        c["wall_s"] = now - self.t_round_start
        if torch.cuda.is_available():
            c["peak_alloc_MiB"] = torch.cuda.max_memory_allocated() / 2**20
            c["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
            torch.cuda.reset_peak_memory_stats()
        c["ga_overhead_s"] = c["ga_s"] - c["fitness_eval_s"]
        c["other_s"] = c["wall_s"] - (c["local_train_s"] + c["qi_s"] + c["agg_s"] + c["ga_s"] + c["ood_eval_s"])
        self.log({"event": "round", **c})
        self.round_times.append(c["wall_s"])
        self.t_round_start = now
        if self.round == 3 and self.total_rounds > 3:
            projected = sum(self.round_times) / len(self.round_times) * self.total_rounds
            self.log({"event": "projection_after_3", "projected_s": projected, "cap_s": self.cap_s})
            if projected > self.cap_s:
                raise StopEarly(f"projected {projected/3600:.2f} h > cap {self.cap_s/3600:.2f} h")
        self.new_round()


def main():
    if "--" in sys.argv:
        i = sys.argv.index("--")
        own, passthrough = sys.argv[1:i], sys.argv[i + 1:]
    else:
        own, passthrough = sys.argv[1:], []
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["breast", "prostate"], required=True)
    ap.add_argument("--mode", choices=["base", "xfedga"], required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cap_hours", type=float, default=3.0)
    args = ap.parse_args(own)

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(REPO, CONFIGS[args.task])) as f:
        cfg = yaml.safe_load(f)
    for c in cfg["clients"]:
        c["path"] = os.path.join(REPO, c["path"])
    cfg["server"]["path"] = os.path.join(REPO, cfg["server"]["path"])
    cfg_path = os.path.join(workdir, "config_abs_paths.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    rounds = 15
    if "--rounds" in passthrough:
        rounds = int(passthrough[passthrough.index("--rounds") + 1])

    import importlib
    mod = importlib.import_module(MODULES[args.task])
    rec = Recorder(os.path.join(workdir, "timing_events.jsonl"), args.cap_hours, rounds)
    rec.log({"event": "start", "task": args.task, "mode": args.mode, "argv": passthrough,
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")})

    mod.local_update = rec.wrap(mod.local_update, "local_train_s", "n_local_updates")
    mod.compute_client_saliency_score = rec.wrap(mod.compute_client_saliency_score, "qi_s")
    mod.fedavg_aggregate = rec.wrap(mod.fedavg_aggregate, "agg_s")
    mod.explanation_weighted_aggregate = rec.wrap(mod.explanation_weighted_aggregate, "agg_s")
    mod.run_ga = rec.wrap(mod.run_ga, "ga_s")
    orig_bfe = mod.build_fitness_evaluator
    mod.build_fitness_evaluator = lambda *a, **kw: rec.wrap_fitness(orig_bfe(*a, **kw))
    mod.evaluate_dsc = rec.wrap(mod.evaluate_dsc, "ood_eval_s")
    # evaluate_explainability is called exactly once per round, last -> round boundary
    mod.evaluate_explainability = rec.wrap(mod.evaluate_explainability, "ood_eval_s", on_end=rec.end_round)

    os.chdir(workdir)
    sys.argv = [mod.__file__, "--config", cfg_path, "--mode", args.mode] + passthrough
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = "completed"
    try:
        mod.main()
    except StopEarly as e:
        status = f"stopped_early: {e}"
        print(f"\n[timed_run] {status}")
    rec.log({"event": "end", "status": status, "total_s": time.time() - rec.t_start,
             "rounds_completed": len(rec.round_times)})


if __name__ == "__main__":
    main()
