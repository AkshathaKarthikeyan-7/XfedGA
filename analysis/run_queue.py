"""
Sequential, resumable experiment queue.

    & $py analysis/run_queue.py queues/beta_screen.yaml
    & $py <frozen-worktree>/analysis/run_queue.py <queue.yaml> --repo "D:\\new repo\\xfedga"

Queue YAML:
    name: beta_screen
    out: results/runs            # relative to --repo (default: current directory)
    runs:
      - {task: prostate, method: xfedga, seed: 1, set: ["aggregation.beta=0.0"], tag: beta0}
      - ...

Behaviour:
  - Code (runner + configs) comes from THIS file's checkout, so the queue can run from a frozen
    git worktree; data ("raw data/"), the preprocessing cache ("cache/") and results come from
    --repo (the working directory of every run).
  - A run whose folder already holds run_meta.json is skipped (finished). A folder without it is an
    interrupted run: it is moved to <out>/_interrupted/ and the run is restarted from scratch
    (runs are seed-deterministic, so a restart reproduces the same result).
  - Each run is a separate process; nvidia-smi is sampled every 10 s into <run>/gpu_state.csv and
    summarised in <run>/gpu_summary.json. A run is flagged THROTTLED if, while the GPU is busy, the
    median SM clock is below --min_clock_mhz or more than 20 % of samples carry a throttle reason.
  - Create the file <out>/STOP to stop cleanly after the current run.
  - Progress: <out>/_queues/<name>.log and <name>_status.json.
"""
import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import yaml

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)

from experiments.config import config_hash, load_config  # noqa: E402

SMI_FIELDS = "timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active,pstate,memory.used"


def run_id_for(entry):
    """Same id as experiments/run_federated.py::run_experiment."""
    cfg = load_config(entry["task"], entry.get("set", []))
    chash = config_hash({"cfg": cfg, "method": entry["method"]})
    tag = entry.get("tag", "")
    return f"{entry['task']}_{entry['method']}_seed{entry['seed']}_{chash[:8]}" + (f"_{tag}" if tag else "")


class GpuSampler(threading.Thread):
    def __init__(self, path, interval=10):
        super().__init__(daemon=True)
        self.path, self.interval, self.stop_event = path, interval, threading.Event()

    def run(self):
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            f.write(SMI_FIELDS + "\n")
            while not self.stop_event.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=15).stdout.strip()
                    if out:
                        f.write(out + "\n")
                        f.flush()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self.stop_event.wait(self.interval)


def gpu_summary(path, min_clock_mhz):
    busy, mem = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                mem.append(float(row.get("memory.used") or "nan"))
            except ValueError:
                pass
            try:
                util = float(row["utilization.gpu"])
                clk = float(row["clocks.sm"])
                pwr = float(row["power.draw"])
                reason = int(row["clocks_event_reasons.active"], 16)
            except (KeyError, ValueError, TypeError):
                continue
            if util >= 50:
                busy.append((clk, pwr, reason))
    max_mem = max((m for m in mem if m == m), default=None)   # whole-GPU memory.used (all processes)
    if not busy:
        return {"busy_samples": 0, "throttled": None, "max_memory_used_MiB": max_mem}
    clocks = sorted(b[0] for b in busy)
    share = sum(1 for b in busy if b[2] & ~0x1) / len(busy)  # any reason other than "GPU idle"
    median = clocks[len(clocks) // 2]
    return {"busy_samples": len(busy), "median_sm_clock_MHz": median, "min_sm_clock_MHz": clocks[0],
            "median_power_W": sorted(b[1] for b in busy)[len(busy) // 2],
            "share_busy_with_throttle_reason": round(share, 3),
            "throttled": median < min_clock_mhz or share > 0.2, "max_memory_used_MiB": max_mem}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue")
    ap.add_argument("--repo", default=os.getcwd(), help="data / cache / results root (working directory of runs)")
    ap.add_argument("--min_clock_mhz", type=float, default=2000.0)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true", help="list run ids and status, run nothing")
    args = ap.parse_args()

    q = yaml.safe_load(open(args.queue, encoding="utf-8"))
    repo = os.path.abspath(args.repo)
    out = os.path.join(repo, q.get("out", "results/runs"))
    qdir = os.path.join(out, "_queues")
    os.makedirs(qdir, exist_ok=True)
    name = q.get("name", os.path.splitext(os.path.basename(args.queue))[0])
    log_path, status_path = os.path.join(qdir, f"{name}.log"), os.path.join(qdir, f"{name}_status.json")
    runner = os.path.join(CODE_ROOT, "experiments", "run_federated.py")
    launcher = os.path.join(CODE_ROOT, "fl_dist", "launch.py")   # entries with runner: fl_dist (parity runs)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=CODE_ROOT).stdout.strip()

    def log(msg):
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    status = {"queue": name, "code_root": CODE_ROOT, "code_commit": commit, "repo": repo, "runs": []}
    log(f"queue {name}: {len(q['runs'])} runs; code {CODE_ROOT} @ {commit}; repo {repo}")
    for i, entry in enumerate(q["runs"], 1):
        rid = run_id_for(entry)
        run_dir = os.path.join(out, rid)
        rec = {"run_id": rid, **entry}
        if os.path.exists(os.path.join(run_dir, "run_meta.json")):
            rec["status"] = "skipped (finished)"
            status["runs"].append(rec)
            log(f"[{i}/{len(q['runs'])}] skip finished {rid}")
            continue
        if args.dry_run:
            rec["status"] = "pending"
            status["runs"].append(rec)
            log(f"[{i}/{len(q['runs'])}] pending {rid}")
            continue
        if os.path.exists(os.path.join(out, "STOP")):
            log("STOP file found; stopping before " + rid)
            break
        if os.path.isdir(run_dir):
            dest = os.path.join(out, "_interrupted", f"{rid}_{dt.datetime.now():%Y%m%d_%H%M%S}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(run_dir, dest)
            log(f"[{i}] moved interrupted run to {dest}")
        os.makedirs(run_dir, exist_ok=True)
        script = launcher if entry.get("runner") == "fl_dist" else runner
        cmd = [args.python, script, "--task", entry["task"], "--method", entry["method"], "--seed", str(entry["seed"]),
               "--out", out] + (["--tag", entry["tag"]] if entry.get("tag") else []) + \
              (["--set"] + list(entry["set"]) if entry.get("set") else []) + (["--no_test"] if entry.get("no_test") else [])
        log(f"[{i}/{len(q['runs'])}] start {rid}")
        sampler = GpuSampler(os.path.join(run_dir, "gpu_state.csv"))
        sampler.start()
        t0 = time.time()
        with open(os.path.join(run_dir, "stdout.log"), "w", encoding="utf-8") as so:
            rc = subprocess.run(cmd, cwd=repo, stdout=so, stderr=subprocess.STDOUT).returncode
        sampler.stop_event.set()
        sampler.join(timeout=30)
        g = gpu_summary(os.path.join(run_dir, "gpu_state.csv"), args.min_clock_mhz)
        with open(os.path.join(run_dir, "gpu_summary.json"), "w", encoding="utf-8") as f:
            json.dump(g, f, indent=1)
        rec.update({"status": "done" if rc == 0 else f"FAILED rc={rc}", "minutes": round((time.time() - t0) / 60, 1),
                    "gpu": g})
        status["runs"].append(rec)
        log(f"[{i}/{len(q['runs'])}] {rec['status']} {rid} in {rec['minutes']} min; GPU median "
            f"{g.get('median_sm_clock_MHz')} MHz" + ("  ** THROTTLED **" if g.get("throttled") else ""))
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=1)
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=1)
    log(f"queue {name} finished")


if __name__ == "__main__":
    main()
