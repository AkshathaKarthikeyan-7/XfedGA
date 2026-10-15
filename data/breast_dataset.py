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

from data.cache import DEFAULT_CACHE_ROOT, build_or_load, file_fingerprint, source_hash


def _load_resized(image_path, mask_path, size=(384, 384)):
    """Read grayscale image and mask, resize to `size`; returns (uint8 image, binary uint8 mask).
    This is the cached part of the preprocessing."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Masks are read in colour and binarised on the brightest channel (audit D-1): BUS-UCLM
    # colour-codes lesions (benign pure green, malignant pure red); a grayscale read maps red to
    # 76 < 127, which silently emptied every malignant mask. White masks (BUSI, BUS-BRA) are unchanged.
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR).max(axis=2)

    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8)  # binarize
    return img, mask


def _normalize(img):
    """Min-max normalize a resized uint8 image to float32 [0,1] (base paper's preprocessing)."""
    img = img.astype(np.float32)
    denom = (img.max() - img.min())
    return (img - img.min()) / denom if denom > 0 else img


def _load_and_preprocess(image_path, mask_path, size=(384, 384)):
    img, mask = _load_resized(image_path, mask_path, size)
    return _normalize(img), mask.astype(np.float32)


def breast_cache(dataset_name, root, size=(384, 384), cache_root=DEFAULT_CACHE_ROOT):
    """Cache of resized uint8 images + binary masks for EVERY pair the indexer finds in `root`.
    Item id = image path. Normalization is applied at load time, so cached and on-the-fly
    outputs are identical."""
    pairs = INDEXERS[dataset_name](root)
    spec = {"dataset": dataset_name, "size": list(size), "version": 1,
            "code": source_hash(_load_resized),
            "files": file_fingerprint([p for pair in pairs for p in pair])}
    items = [(img_path, (img_path, mask_path)) for img_path, mask_path in pairs]
    return build_or_load(dataset_name, spec, items,
                         lambda pm: [_load_resized(pm[0], pm[1], tuple(size))],
                         cache_root=cache_root, image_dtype=np.uint8)


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
    """(image, mask) pairs. With `cache_root` set (default), reads preprocessed arrays from the
    on-disk cache (see breast_cache), built on first use; with cache_root=None, reads from disk."""

    def __init__(self, client_name, root, size=(384, 384), pairs=None, cache_root=DEFAULT_CACHE_ROOT):
        self.size = size
        self.pairs = pairs if pairs is not None else INDEXERS[client_name](root)
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No (image, mask) pairs found for {client_name} at {root}. "
                f"Check the folder structure / update the indexer function."
            )
        self.cache = breast_cache(client_name, root, size, cache_root) if cache_root else None
        self.rows = [self.cache.rows(p)[0] for p, _ in self.pairs] if self.cache else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache is not None:
            row = self.rows[idx]
            img = _normalize(self.cache.image(row))
            mask = self.cache.mask(row)
        else:
            img_path, mask_path = self.pairs[idx]
            img, mask = _load_and_preprocess(img_path, mask_path, self.size)
        img_t = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(np.asarray(mask)).long()        # (H, W)
        return img_t, mask_t


def train_val_split(pairs, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs))
    n_val = int(len(pairs) * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]


def filter_nonempty_masks(pairs, size=(384, 384), min_fg_fraction=0.0, verbose=True, cache=None):
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
        if cache is not None:
            fg_frac = cache.mask_sum(cache.rows(img_path)[0]) / (size[0] * size[1])
        else:
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