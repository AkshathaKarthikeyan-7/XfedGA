"""
Dataset loader for breast ultrasound segmentation clients (BUSI, BUS-BRA, BUS-UCLM).

Each dataset has a slightly different folder layout, so this module normalizes
them into a common (image, mask) pair list before wrapping in a torch Dataset.
Update `_index_busi` / `_index_busbra` / `_index_busuclm` once you've unzipped
the actual files and can see the real folder structure.
"""
import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_and_preprocess(image_path, mask_path, size=(384, 384)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

    # min-max normalize image to [0,1], matches base paper's preprocessing
    img = img.astype(np.float32)
    denom = (img.max() - img.min())
    img = (img - img.min()) / denom if denom > 0 else img
    mask = (mask > 127).astype(np.float32)  # binarize

    return img, mask


def _index_busi(root):
    """
    BUSI layout (Kaggle 'aryashah2k' version):
        root/benign/*.png (+ *_mask.png, some have _mask_1.png for multiple lesions)
        root/malignant/*.png (+ *_mask.png)
        root/normal/*.png (+ *_mask.png, usually all-zero mask)
    Returns list of (image_path, mask_path) tuples, skipping normal images
    with empty masks unless you want to keep them as negative examples.
    """
    pairs = []
    for cls in ["benign", "malignant", "normal"]:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        images = sorted(f for f in glob.glob(os.path.join(cls_dir, "*.png")) if "_mask" not in f)
        for img_path in images:
            base = img_path[:-4]
            mask_path = base + "_mask.png"
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path))
    return pairs


def _index_busbra(root):
    """
    BUS-BRA layout (Kaggle 'orvile' upload), confirmed structure:
        root/BUSBRA/Images/bus_0001-l.png
        root/BUSBRA/Masks/mask_0001-l.png   (bus_ prefix swapped for mask_,
                                              suffix/number otherwise identical)
        root/BUSBRA/bus_data.csv            (per-image scanner metadata --
                                              useful later for further
                                              sub-splitting by scanner)
    `root` should point at the BUS-BRA folder itself (e.g. "raw data/BUS-BRA");
    this function looks inside its BUSBRA/ subfolder automatically.
    """
    base = os.path.join(root, "BUSBRA")
    if not os.path.isdir(base):
        base = root  # fallback in case the nesting differs on your machine

    images_dir = os.path.join(base, "Images")
    masks_dir = os.path.join(base, "Masks")
    pairs = []
    if not (os.path.isdir(images_dir) and os.path.isdir(masks_dir)):
        return pairs

    for img_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
        fname = os.path.basename(img_path)
        mask_fname = fname.replace("bus_", "mask_", 1)  # bus_0001-l.png -> mask_0001-l.png
        mask_path = os.path.join(masks_dir, mask_fname)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            # fallback to the more permissive matcher in case a filename
            # doesn't follow the bus_/mask_ convention exactly
            stem, ext = os.path.splitext(fname)
            fallback = _find_matching_mask(masks_dir, stem, ext)
            if fallback:
                pairs.append((img_path, fallback))
    return pairs


def _find_matching_mask(masks_dir, stem, ext):
    """Tries a few common mask-naming conventions; extend this list if none match."""
    candidates = [
        f"{stem}{ext}",            # exact same filename (most likely for BUS-BRA)
        f"{stem}_mask{ext}",       # BUSI-style suffix, just in case
        f"{stem}-mask{ext}",
    ]
    for c in candidates:
        p = os.path.join(masks_dir, c)
        if os.path.exists(p):
            return p
    return None


def _index_busuclm(root):
    """
    BUS-UCLM layout (Kaggle 'orvile' upload), confirmed structure:
        root/BUS-UCLM Breast ultrasound lesion segmentation dataset/
             BUS-UCLM Breast ultrasound lesion segmentation dataset/
             BUS-UCLM/images/ALWI_000.png
             BUS-UCLM/masks/ALWI_000.png    (identical filename in both --
                                              simplest case of the three
                                              breast datasets)
        (also ships an alternate 'bus_uclm_separated' class-folder layout
        like BUSI, and an INFO.csv with per-image metadata -- neither is
        needed for pairing since filenames match exactly)

    The nesting above is doubled/inconsistent depending on how the zip was
    extracted, so rather than hardcode the exact depth, we walk the tree
    looking for a directory literally named "images" that has a sibling
    "masks" directory -- robust to whatever depth it lands at.
    """
    for dirpath, dirnames, _ in os.walk(root):
        if "images" in dirnames and "masks" in dirnames:
            images_dir = os.path.join(dirpath, "images")
            masks_dir = os.path.join(dirpath, "masks")
            pairs = []
            for img_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
                fname = os.path.basename(img_path)
                mask_path = os.path.join(masks_dir, fname)  # identical filename
                if os.path.exists(mask_path):
                    pairs.append((img_path, mask_path))
            if pairs:
                return pairs
    return []


INDEXERS = {
    "BUSI": _index_busi,
    "BUS-BRA": _index_busbra,
    "BUS-UCLM": _index_busuclm,
}


class BreastClientDataset(Dataset):
    def __init__(self, client_name, root, size=(384, 384), pairs=None):
        self.size = size
        self.pairs = pairs if pairs is not None else INDEXERS[client_name](root)
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found for {client_name} at {root}. "
                f"Check the folder structure / update the indexer function."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img, mask = _load_and_preprocess(img_path, mask_path, self.size)
        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask).long()        # (H, W)
        return img_t, mask_t


def train_val_split(pairs, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(len(pairs) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def filter_nonempty_masks(pairs, size=(384, 384), min_fg_fraction=0.0, verbose=True):
    """
    Removes (image, mask) pairs whose mask has no foreground (or below
    min_fg_fraction of pixels), to prevent the model learning an
    "always predict background" shortcut on datasets that mix in
    no-lesion ("Normal") images alongside real lesion images.

    min_fg_fraction=0.0 (default) only removes truly empty masks;
    raise it slightly (e.g. 0.001) to also drop near-empty/noise masks.
    """
    kept = []
    dropped = 0
    for img_path, mask_path in pairs:
        _, mask = _load_and_preprocess(img_path, mask_path, size)
        fg_frac = mask.sum() / mask.size
        if fg_frac > min_fg_fraction:
            kept.append((img_path, mask_path))
        else:
            dropped += 1
    if verbose:
        print(f"  filter_nonempty_masks: kept {len(kept)}/{len(pairs)} "
              f"(dropped {dropped} empty/near-empty masks)")
    return kept