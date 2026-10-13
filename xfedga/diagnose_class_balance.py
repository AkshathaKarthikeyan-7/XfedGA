"""
Checks foreground (lesion) pixel percentage per client -- helps diagnose
whether a client's masks are dominated by empty/near-empty images, which can
cause the model to collapse to an "always predict background" shortcut.

Samples RANDOMLY across the full dataset (not just the first N), since
indexers built from class-organized folders (e.g. BUSI's benign/malignant/
normal) list images in a fixed order, and a sequential sample can badly
under- or over-represent a class.

Usage:
    python diagnose_class_balance.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from data.breast_dataset import BreastClientDataset

RNG = np.random.RandomState(42)

datasets = {
    "BUSI": "raw data/BUSI",
    "BUS-UCLM": "raw data/BUS-UCLM",
    "BUS-BRA": "raw data/BUS-BRA",
}

for name, path in datasets.items():
    ds = BreastClientDataset(name, path)
    n_check = min(len(ds), 300)
    indices = RNG.choice(len(ds), size=n_check, replace=False)  # random, not sequential

    fg_fractions = []
    empty_count = 0
    for i in indices:
        _, mask = ds[i]
        mask_np = mask.numpy()
        fg_frac = mask_np.sum() / mask_np.size
        fg_fractions.append(fg_frac)
        if fg_frac == 0:
            empty_count += 1

    fg_fractions = np.array(fg_fractions)
    print(f"\n{name} (random sample {n_check}/{len(ds)}):")
    print(f"  empty masks: {empty_count}/{n_check} ({100*empty_count/n_check:.1f}%)")
    print(f"  mean foreground pixel %: {100*fg_fractions.mean():.3f}%")
    print(f"  median foreground pixel %: {100*np.median(fg_fractions):.3f}%")