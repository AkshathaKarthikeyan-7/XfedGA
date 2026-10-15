"""Preprocessing cache: exact equality with on-the-fly preprocessing, keying, and worker safety."""
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from data.cache import build_or_load, cache_key
from tests.conftest import require_path

BREAST = [("BUSI", "raw data/BUSI"), ("BUS-UCLM", "raw data/BUS-UCLM"), ("BUS-BRA", "raw data/BUS-BRA")]
PROSTATE = [("PROMISE12", "raw data/PROMISE12"), ("Decathlon", "raw data/Task05_Prostate/Task05_Prostate")]


@pytest.mark.data
@pytest.mark.parametrize("name,root", BREAST)
def test_breast_cache_equals_on_the_fly(name, root):
    require_path(root)
    from data.breast_dataset import BreastClientDataset, INDEXERS
    pairs = INDEXERS[name](root)
    pick = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 6).astype(int)]
    cached = BreastClientDataset(name, root, pairs=pick)
    direct = BreastClientDataset(name, root, pairs=pick, cache_root=None)
    for i in range(len(pick)):
        (xc, yc), (xd, yd) = cached[i], direct[i]
        assert xc.dtype == xd.dtype and torch.equal(xc, xd)
        assert torch.equal(yc, yd)


@pytest.mark.data
@pytest.mark.parametrize("name,root", PROSTATE)
def test_prostate_cache_equals_on_the_fly(name, root):
    require_path(root)
    from data.prostate_dataset import INDEXERS, ProstateClientDataset
    cases = INDEXERS[name](root)
    pick = [cases[0], cases[len(cases) // 2]]
    cached = ProstateClientDataset(name, root, cases=pick)
    direct = ProstateClientDataset(name, root, cases=pick, cache_root=None)
    assert len(cached) == len(direct) > 0
    for i in range(len(direct)):
        (xc, yc), (xd, yd) = cached[i], direct[i]
        assert torch.equal(xc, xd)
        assert torch.equal(yc, yd)


def test_cache_key_changes_with_spec():
    base = {"dataset": "x", "size": [384, 384], "code": "abc", "files": [["a", 1, 2]]}
    assert cache_key(base) == cache_key(dict(base))
    for change in [{"size": [256, 256]}, {"code": "abd"}, {"files": [["a", 1, 3]]}]:
        assert cache_key({**base, **change}) != cache_key(base)


def _toy_cache(tmp_path, n=12):
    rng = np.random.default_rng(0)
    items = [(f"item{i:02d}", i) for i in range(n)]
    return build_or_load("toy", {"n": n}, items,
                         lambda i: [(rng.random((8, 8)).astype(np.float32), np.eye(8) * (i % 2))],
                         cache_root=str(tmp_path))


def test_build_then_load_is_identical(tmp_path):
    a = _toy_cache(tmp_path)
    b = build_or_load("toy", {"n": 12}, [], lambda i: [], cache_root=str(tmp_path))  # must load, not rebuild
    assert a.directory == b.directory
    for r in range(12):
        assert np.array_equal(a.image(r), b.image(r)) and np.array_equal(a.mask(r), b.mask(r))


def test_cached_arrays_pickle_without_data(tmp_path):
    c = _toy_cache(tmp_path)
    c.image(0)  # open the memmaps
    blob = pickle.dumps(c)
    assert len(blob) < 4096
    c2 = pickle.loads(blob)
    assert np.array_equal(c2.image(3), c.image(3))


class _ToyDataset(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache.index)

    def __getitem__(self, i):
        return torch.from_numpy(self.cache.image(i)), torch.from_numpy(self.cache.mask(i)).long()


def test_dataloader_workers_match_single_process(tmp_path):
    ds = _ToyDataset(_toy_cache(tmp_path))
    single = [b for b in DataLoader(ds, batch_size=4, num_workers=0)]
    multi = [b for b in DataLoader(ds, batch_size=4, num_workers=2, persistent_workers=True)]
    assert len(single) == len(multi)
    for (x0, y0), (x1, y1) in zip(single, multi):
        assert torch.equal(x0, x1) and torch.equal(y0, y1)


@pytest.mark.data
def test_relative_and_absolute_roots_share_one_cache():
    import os
    root = "raw data/PROMISE12"
    require_path(root)
    from data.prostate_dataset import prostate_cache
    assert prostate_cache("PROMISE12", root).directory == prostate_cache("PROMISE12", os.path.abspath(root)).directory
