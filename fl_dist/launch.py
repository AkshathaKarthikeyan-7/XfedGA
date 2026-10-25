"""
One-command launcher for the multi-process federation on a single machine over loopback gRPC.

    & $py fl_dist/launch.py --task breast --method xfedga --seed 1
    & $py fl_dist/launch.py --task prostate --method xfedga --seed 1 --set train.rounds=1 --no_test   # smoke

Starts the server, waits until it listens, then one OS process per client. Logs per process in
<out>/<run_id>/{server,client_1,client_2}.log, nvidia-smi samples in gpu_state.csv (peak memory
with all processes running). Ctrl+C, or any process exiting with an error, terminates all of them.
Only the server address (127.0.0.1:<free port> here) would change across machines; not tested.
"""
import argparse
import os
import socket
import subprocess
import sys
import time

CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_ROOT)

from analysis.run_queue import GpuSampler, gpu_summary  # noqa: E402
from experiments.config import config_hash, load_config  # noqa: E402


def free_port():
    """A free local TCP port for the gRPC server."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_listening(port, proc, timeout=300):
    """Wait until the server listens on `port` (False if it exits or times out)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False


def terminate(procs):
    """Terminate the processes, then kill those still alive after 20 s."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.time() + 20
    for p in procs:
        try:
            p.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    """Start the server and one process per client; wait for them; write the GPU summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["breast", "prostate", "prostate_ms", "prostate_ms_runmc"], required=True)
    ap.add_argument("--method", default="xfedga")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="fl")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--no_test", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    cfg = load_config(args.task, args.set)
    chash = config_hash({"cfg": cfg, "method": args.method})
    run_id = f"{args.task}_{args.method}_seed{args.seed}_{chash[:8]}_{args.tag}"
    out_dir = os.path.abspath(os.path.join(args.out, run_id))
    os.makedirs(out_dir, exist_ok=True)
    port = free_port()
    address = f"127.0.0.1:{port}"
    common = ["--task", args.task, "--method", args.method, "--seed", str(args.seed)] + \
             (["--set"] + list(args.set) if args.set else [])
    sampler = GpuSampler(os.path.join(out_dir, "gpu_state.csv"), interval=5)
    sampler.start()
    procs, logs = [], []
    print(f"[launch] run {run_id} on {address}; logs in {out_dir}", flush=True)
    try:
        f = open(os.path.join(out_dir, "server.log"), "w", encoding="utf-8")
        logs.append(f)
        server = subprocess.Popen([args.python, os.path.join(CODE_ROOT, "fl_dist", "server.py"), *common,
                                   "--server_address", address, "--out_dir", out_dir] +
                                  (["--no_test"] if args.no_test else []), stdout=f, stderr=subprocess.STDOUT)
        procs.append(server)
        if not wait_listening(port, server):
            raise RuntimeError("server did not start listening; see server.log")
        print(f"[launch] server listening (pid {server.pid})", flush=True)
        for k in range(len(cfg["clients"])):
            f = open(os.path.join(out_dir, f"client_{k + 1}.log"), "w", encoding="utf-8")
            logs.append(f)
            procs.append(subprocess.Popen([args.python, os.path.join(CODE_ROOT, "fl_dist", "client.py"), *common,
                                           "--client", str(k), "--server_address", address,
                                           "--audit_out", os.path.join(out_dir, f"client_{k + 1}_audit.json"),
                                           "--gpu_lock", os.path.join(out_dir, "gpu.lock")],
                                          stdout=f, stderr=subprocess.STDOUT))
            print(f"[launch] client {k + 1} started (pid {procs[-1].pid})", flush=True)
        while True:
            codes = [p.poll() for p in procs]
            if any(c not in (None, 0) for c in codes):
                failed = [i for i, c in enumerate(codes) if c not in (None, 0)]
                raise RuntimeError(f"process(es) {failed} exited with {[codes[i] for i in failed]}; see logs")
            if all(c == 0 for c in codes):
                break
            time.sleep(2)
        print("[launch] all processes finished cleanly", flush=True)
        rc = 0
    except KeyboardInterrupt:
        print("[launch] Ctrl+C: stopping all processes", flush=True)
        rc = 130
    except RuntimeError as e:
        print(f"[launch] ERROR: {e}; stopping all processes", flush=True)
        rc = 1
    finally:
        terminate(procs)
        for f in logs:
            f.close()
        sampler.stop_event.set()
        sampler.join(timeout=30)
        g = gpu_summary(os.path.join(out_dir, "gpu_state.csv"), 2000.0)
        print(f"[launch] GPU: {g}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
