"""
On-disk cache of preprocessed (image, mask) arrays, built once per dataset.

Each cache lives in ``<cache_root>/<dataset>/<key>/`` with:
    images.npy   (N, H, W) array (dtype chosen by the caller)
    masks.npy    (N, H, W) uint8, binary {0, 1}
    index.json   {item_id: [start_row, end_row)} -- one item can own several rows
                 (a 3-D case owns one row per kept slice)
    spec.json    the preprocessing spec that produced the cache

The key is a SHA-256 over the preprocessing spec: preprocessing parameters,
the source code of the preprocessing functions, and (path, size, mtime) of
every input file. Any change to the code, parameters or raw files produces a
new key, so a stale cache is never read.

Arrays are opened lazily with ``mmap_mode="r"`` inside each process, so
datasets that hold a ``CachedArrays`` can be pickled into DataLoader worker
processes (Windows spawn) without copying the data.
"""
import hashlib
import inspect
import json
import logging
import os
import shutil
import uuid

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = "cache"


def source_hash(*functions):
    """SHA-256 of the source code of the given functions (part of the cache key)."""
    h = hashlib.sha256()
    for fn in functions:
        h.update(inspect.getsource(fn).encode("utf-8"))
    return h.hexdigest()


def item_key(path):
    """Canonical item id: absolute, case-normalised path (relative and absolute roots share a cache)."""
    return os.path.normcase(os.path.abspath(path))


def file_fingerprint(paths):
    """(absolute path, size, mtime_ns) for every input file, in a stable order."""
    out = []
    for p in sorted({item_key(p) for p in paths}):
        st = os.stat(p)
        out.append([p, st.st_size, st.st_mtime_ns])
    return out


def cache_key(spec):
    """Short SHA-256 of the preprocessing spec (code, parameters and file fingerprints)."""
    blob = json.dumps(spec, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class CachedArrays:
    """Lazily memory-mapped view of one cache directory; safe to pickle."""

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "index.json")) as f:
            self.index = {k: tuple(v) for k, v in json.load(f).items()}
        self._images = None
        self._masks = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        state["_masks"] = None
        return state

    def _open(self):
        if self._images is None:
            self._images = np.load(os.path.join(self.directory, "images.npy"), mmap_mode="r")
            self._masks = np.load(os.path.join(self.directory, "masks.npy"), mmap_mode="r")

    def rows(self, item_id):
        """Row indices of `item_id` in the cache."""
        start, end = self.index[item_key(item_id)]
        return list(range(start, end))

    def image(self, row):
        """Preprocessed image of one row."""
        self._open()
        return np.array(self._images[row])

    def mask(self, row):
        """Binary mask of one row."""
        self._open()
        return np.array(self._masks[row])

    def mask_sum(self, row):
        """Number of foreground pixels in the mask of one row."""
        self._open()
        return int(self._masks[row].sum())


def build_or_load(dataset, spec, items, preprocess_item, cache_root=DEFAULT_CACHE_ROOT, image_dtype=np.float32):
    """
    Return CachedArrays for ``items`` under ``spec``; build the cache if missing.

    items:            list of (item_id, payload); item_id is a file path (see item_key);
                      payload is passed to preprocess_item
    preprocess_item:  payload -> list of (image HxW, mask HxW) arrays
    """
    key = cache_key(spec)
    directory = os.path.join(cache_root, dataset, key)
    if os.path.isfile(os.path.join(directory, "index.json")):
        return CachedArrays(directory)

    log.info("building cache %s (%d items)", directory, len(items))
    images, masks, index = [], [], {}
    for item_id, payload in items:
        start = len(images)
        for img, msk in preprocess_item(payload):
            images.append(np.asarray(img, dtype=image_dtype))
            masks.append((np.asarray(msk) > 0).astype(np.uint8))
        index[item_key(item_id)] = [start, len(images)]

    tmp = os.path.join(cache_root, dataset, f".tmp-{uuid.uuid4().hex}")
    os.makedirs(tmp)
    np.save(os.path.join(tmp, "images.npy"), np.stack(images))
    np.save(os.path.join(tmp, "masks.npy"), np.stack(masks))
    with open(os.path.join(tmp, "index.json"), "w") as f:
        json.dump(index, f)
    with open(os.path.join(tmp, "spec.json"), "w") as f:
        json.dump(spec, f, indent=1, default=str)
    try:
        os.replace(tmp, directory)
    except OSError:
        # another process finished the same cache first; keep theirs
        shutil.rmtree(tmp, ignore_errors=True)
    return CachedArrays(directory)
