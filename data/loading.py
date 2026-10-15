"""DataLoader options from the task YAML (`loader:` section).

num_workers defaults to 0: measured 15 Oct 2026 on the cached datasets (RTX 4050 laptop,
Windows), a steady training epoch took the same time with 0, 2 or 4 workers
(prostate 1.96/1.99/2.00 s, BUSI 2.68/2.70/2.73 s), while worker start-up under
Windows spawn added 8-30 s per loader. Workers > 0 are supported and tested
(tests/test_cache.py::test_dataloader_workers_match_single_process).
"""


def loader_kwargs(cfg):
    opts = (cfg or {}).get("loader", {}) or {}
    workers = int(opts.get("num_workers", 0))
    return {
        "num_workers": workers,
        "persistent_workers": bool(opts.get("persistent_workers", True)) and workers > 0,
        "pin_memory": bool(opts.get("pin_memory", False)),
    }
