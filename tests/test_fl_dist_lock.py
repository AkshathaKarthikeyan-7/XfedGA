"""
fl_dist cross-process GPU lock (fl_dist/gpu_lock.py; fix of the 5-client OOM, 18 Oct 2026).
  - fast: competing processes never hold the lock at the same time;
  - slow, CPU only (GPU hidden): a tiny 5-client prostate_ms fl_dist run completes, every client logs its lock
    use, and the message audit still shows exactly weights, n_i, q_i and the client index.
"""
import json
import os
import subprocess
import sys

import pytest

from tests.conftest import REPO, require_path

HOLDER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from fl_dist.gpu_lock import gpu_lock
with gpu_lock(sys.argv[2]) as waited:
    t0 = time.time(); time.sleep(0.4); t1 = time.time()
print(json.dumps({"start": t0, "end": t1, "waited": waited}))
"""


def test_lock_serialises_processes(tmp_path):
    lock = str(tmp_path / "gpu.lock")
    procs = [subprocess.Popen([sys.executable, "-c", HOLDER, REPO, lock], stdout=subprocess.PIPE, text=True)
             for _ in range(3)]
    spans = sorted((json.loads(p.communicate(timeout=60)[0]) for p in procs), key=lambda s: s["start"])
    assert all(p.returncode == 0 for p in procs)
    for a, b in zip(spans, spans[1:]):
        assert a["end"] <= b["start"] + 1e-3, spans         # no two holders overlap
    assert sum(s["waited"] > 0.2 for s in spans) >= 2        # the others waited


def test_lock_disabled_without_path():
    from fl_dist.gpu_lock import gpu_lock
    with gpu_lock(None) as waited:
        assert waited == 0.0


@pytest.mark.data
@pytest.mark.slow
def test_five_client_fl_dist_on_cpu(tmp_path):
    require_path("raw data/MultiSiteProstate")
    from tests.test_switches import TINY
    out = str(tmp_path / "fl")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, os.path.join(REPO, "fl_dist", "launch.py"), "--task", "prostate_ms",
                        "--method", "xfedga", "--seed", "5", "--no_test", "--out", out, "--set", *TINY],
                       capture_output=True, text=True, timeout=3600, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    run = next(os.path.join(out, d) for d in os.listdir(out))
    audits = [json.load(open(os.path.join(run, f"client_{k}_audit.json"), encoding="utf-8")) for k in range(1, 6)]
    assert all(len(a["gpu_lock"]) == 1 for a in audits)                     # 1 round, lock taken once per client
    audit = json.load(open(os.path.join(run, "message_audit.json"), encoding="utf-8"))
    assert audit["metric_keys_seen"] == ["client_index", "q_i"]
    assert audit["all_shapes_ok"] and audit["dtypes_seen"] == ["float32"] and audit["config_keys_sent"] == ["round"]
