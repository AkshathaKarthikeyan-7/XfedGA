"""
Batch-size memory check: ONE forward + backward pass (plus one Adam step) of the
Residual U-Net with DiceCE loss at 384x384, for several batch sizes. Not a
training run. Answers: does the paper's batch 128 fit in 6 GB?

Writes results/timing/batch_memcheck.json.
Usage (from repo root):
    python analysis/batch_memcheck.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402

from models.residual_unet import build_residual_unet  # noqa: E402
from fedga.losses import build_loss  # noqa: E402

BATCH_SIZES = [8, 16, 32, 64, 96, 128]


def one_step(bs, device):
    model = build_residual_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    loss_fn = build_loss()
    x = torch.rand(bs, 1, 384, 384, device=device)
    y = (torch.rand(bs, 1, 384, 384, device=device) > 0.9).long()
    torch.cuda.reset_peak_memory_stats()
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20, torch.cuda.max_memory_reserved() / 2**20


def main():
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    out = {"gpu": torch.cuda.get_device_name(0), "total_MiB": total, "image": [384, 384], "results": []}
    for bs in BATCH_SIZES:
        try:
            alloc, reserved = one_step(bs, device)
            # On Windows (WDDM) the driver can spill past dedicated VRAM into shared system
            # memory instead of raising OOM, so "no OOM" is not "fits": compare to total VRAM.
            rec = {"batch": bs, "no_oom": True, "peak_alloc_MiB": alloc, "peak_reserved_MiB": reserved,
                   "fits_dedicated_vram": reserved <= total}
        except torch.OutOfMemoryError as e:
            rec = {"batch": bs, "no_oom": False, "fits_dedicated_vram": False,
                   "error": str(e).splitlines()[0][:200]}
        torch.cuda.empty_cache()
        print(json.dumps(rec))
        out["results"].append(rec)
    with open(os.path.join(REPO, "results", "timing", "batch_memcheck.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
