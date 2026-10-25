"""
Data-side scenario switches (configs: `scenario:` section).

- Heterogeneous PROMISE12 partition: clients split by in-plane acquisition spacing read from the
  .mhd headers. PROMISE12 on disk has no site/vendor metadata (audit section 3); the header
  spacing forms three contiguous blocks (fine 0.27-0.35 mm, mid 0.39 mm, coarse 0.47-0.75 mm).
  Client 1 = fine + mid (spacing < threshold), client 2 = coarse. Described as
  "acquisition-profile groups", never as sites.
- Corrupted-label client: every image of that client is paired with the mask of a different
  image of the same client (a fixed, seeded derangement), so its labels are plausible-looking
  but wrong.
"""
import numpy as np
from torch.utils.data import Dataset

from data.prostate_dataset import _index_promise12


def promise12_in_plane_spacing(mhd_path):
    """In-plane voxel spacing (mm) read from a PROMISE12 .mhd header."""
    with open(mhd_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("ElementSpacing"):
                return float(line.split("=")[1].split()[0])
    raise ValueError(f"no ElementSpacing in {mhd_path}")


def split_promise12_by_acquisition(root, threshold_mm=0.45):
    """Returns (client1_cases, client2_cases): spacing < threshold_mm vs >= threshold_mm."""
    cases = _index_promise12(root)
    fine = [c for c in cases if promise12_in_plane_spacing(c[0]) < threshold_mm]
    coarse = [c for c in cases if promise12_in_plane_spacing(c[0]) >= threshold_mm]
    return fine, coarse


def _derangement(n, rng):
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    for i in fixed:  # break fixed points so no image keeps its own mask
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


class ShuffledMaskDataset(Dataset):
    """
    Wraps a dataset so item i returns (image_i, mask_perm[i]).
    fraction = 1.0: perm is a seeded derangement of all items (the original scenario, unchanged).
    fraction p < 1: a seeded subset of round(p * n) items (at least 2) is chosen and their masks are deranged
    among themselves; every other item keeps its own mask (dose-response, master plan item 5).
    """

    def __init__(self, dataset, seed, fraction=1.0):
        self.dataset = dataset
        n = len(dataset)
        rng = np.random.RandomState(seed)
        if fraction >= 1.0:
            perm = _derangement(n, rng)
        else:
            k = int(round(fraction * n))
            if k < 2:
                raise ValueError(f"corruption fraction {fraction} of {n} items corrupts fewer than 2")
            sub = np.sort(rng.permutation(n)[:k])
            perm = np.arange(n)
            perm[sub] = sub[_derangement(k, rng)]
        self.perm = perm
        self.corrupted = np.where(perm != np.arange(n))[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        _, mask = self.dataset[int(self.perm[idx])]
        return img, mask
