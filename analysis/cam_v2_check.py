"""
CAM / EC pipeline check for protocol v2 inputs (18 Oct 2026). CPU only (CUDA hidden), no test data.
  1. Synthetic known-answer variant: Residual U-Net trained briefly on dark noise + one bright square, small
     (~3-8 % of the image, as tests/test_xai_known_answer.py) or gland-sized (~10-19 %, as after the v2 crop),
     with [0,1] or z-scored intensities; full-res CAM layer, "pred" target, EC threshold 0.5.
  2. Real validation slices (Decathlon validation set, 6 fixed indices): v1 vs v2 prostate XFedGA seed 1,
     validation-selected checkpoints: CAM range, share of pixels with CAM >= 0.5, EC, pointing game, CAM mass in GT.
Writes results/diagnostics/cam_v2_check.json.
    & $py analysis/cam_v2_check.py
"""
import glob
import json
import os
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # "" does not hide the GPU on this machine; -1 does
warnings.filterwarnings("ignore")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.config import load_config, seed_everything  # noqa: E402
from experiments.tasks import build_task  # noqa: E402
from fedga.losses import build_loss  # noqa: E402
from models.residual_unet import build_residual_unet  # noqa: E402
from xai.explainability import (explanation_consistency, get_gradcam, pointing_game_accuracy,  # noqa: E402
                                unet_fullres_cam_layer)

assert not torch.cuda.is_available(), "this check must run on the CPU"
dev = torch.device("cpu")
SIZE = 64
IDX = [5, 20, 40, 60, 80, 100]


def sample(rng, n, side, zscore):
    x = rng.normal(0.2, 0.05, size=(n, 1, SIZE, SIZE)).astype(np.float32)
    y = np.zeros((n, SIZE, SIZE), dtype=np.int64)
    for i in range(n):
        s = rng.integers(*side)
        r, c = rng.integers(4, SIZE - s - 4, size=2)
        x[i, 0, r:r + s, c:c + s] += 0.6
        y[i, r:r + s, c:c + s] = 1
    x = np.clip(x, 0, 1)
    if zscore:
        x = (x - x.mean()) / x.std()
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)


def synthetic(side, zscore):
    seed_everything(0)
    rng = np.random.default_rng(0)
    m = build_residual_unet().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    lf = build_loss()
    for _ in range(250):
        x, y = sample(rng, 8, side, zscore)
        opt.zero_grad()
        lf(m(x), y.unsqueeze(1)).backward()
        opt.step()
    m.eval()
    x, y = sample(np.random.default_rng(123), 8, side, zscore)
    with torch.no_grad():
        p = torch.argmax(m(x), 1).numpy()
    out = {"dice": [], "pg": [], "ec": [], "mass": [], "hot_frac": [], "gt_frac": []}
    for i in range(len(y)):
        cam = get_gradcam(m, unet_fullres_cam_layer(m), x[i:i + 1])
        g = y[i].numpy()
        out["dice"].append(2 * (p[i] * g).sum() / (p[i].sum() + g.sum()))
        out["pg"].append(pointing_game_accuracy(cam, g))
        out["ec"].append(explanation_consistency(cam, g, 0.5))
        out["mass"].append(cam[g > 0].sum() / max(cam.sum(), 1e-9))
        out["hot_frac"].append((cam >= 0.5).mean())
        out["gt_frac"].append(g.mean())
    return {k: round(float(np.mean(v)), 3) for k, v in out.items()}


def real(proto, run_glob):
    ov = ["preprocessing.fov_crop_mm=100", "preprocessing.intensity=zscore"] if proto == "v2" else []
    val = build_task("prostate", load_config("prostate", ov), part="server").server["validation"].dataset
    d = glob.glob(run_glob)[0]
    m = build_residual_unet().to(dev)
    m.load_state_dict(torch.load(glob.glob(d + "/*_valselected.pt")[0], map_location=dev))
    m.eval()
    rows = []
    for i in IDX:
        x, g = val[i]
        x, g = x[None], g.numpy()
        with torch.no_grad():
            p = torch.argmax(m(x), 1)[0].numpy()
        cam = get_gradcam(m, unet_fullres_cam_layer(m), x)
        rows.append({"x_range": [round(float(x.min()), 2), round(float(x.max()), 2)],
                     "cam_range": [round(float(cam.min()), 2), round(float(cam.max()), 2)],
                     "gt_frac": round(float(g.mean()), 3), "pred_frac": round(float(p.mean()), 3),
                     "hot_frac": round(float((cam >= 0.5).mean()), 3),
                     "ec": round(explanation_consistency(cam, g, 0.5), 3), "pg": float(pointing_game_accuracy(cam, g)),
                     "mass_in_gt": round(float(cam[g > 0].sum() / max(cam.sum(), 1e-9)), 3),
                     "dsc": round(float(2 * (p * g).sum() / max(p.sum() + g.sum(), 1)), 3)})
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3)
            for k in ["gt_frac", "pred_frac", "hot_frac", "ec", "pg", "mass_in_gt", "dsc"]}
    return {"run": os.path.basename(d), "indices": IDX, "slices": rows, "mean": mean}


def main():
    res = {"synthetic": {}, "real": {}}
    for side, label in [((10, 18), "small square ~3-8%"), ((20, 28), "gland-sized square ~10-19%")]:
        for z in (False, True):
            res["synthetic"][f"{label}, {'z-scored' if z else '[0,1]'}"] = synthetic(side, z)
    res["real"]["v1"] = real("v1", "results/runs/prostate_xfedga_seed1_????????")
    res["real"]["v2"] = real("v2", "results/runs/prostate_xfedga_seed1_2a059f3e_v2screen")
    os.makedirs(os.path.join("results", "diagnostics"), exist_ok=True)
    with open(os.path.join("results", "diagnostics", "cam_v2_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["synthetic"], indent=1))
    print(json.dumps({k: v["mean"] for k, v in res["real"].items()}, indent=1))


if __name__ == "__main__":
    main()
