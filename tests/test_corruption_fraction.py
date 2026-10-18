"""Dose-response label corruption (master plan item 5): ShuffledMaskDataset with a corruption fraction p."""
import numpy as np
import pytest
from torch.utils.data import Dataset

from data.scenarios import ShuffledMaskDataset


class _Ids(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i, i            # (image id, mask id)


def _old_perm(n, seed):    # the pre-change implementation, copied verbatim
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    fixed = np.where(perm == np.arange(n))[0]
    for i in fixed:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]
    return perm


@pytest.mark.parametrize("n,seed", [(10, 1), (237, 778), (300, 780)])
def test_fraction_one_is_the_original_scenario(n, seed):
    assert np.array_equal(ShuffledMaskDataset(_Ids(n), seed).perm, _old_perm(n, seed))
    assert np.array_equal(ShuffledMaskDataset(_Ids(n), seed, 1.0).perm, _old_perm(n, seed))


@pytest.mark.parametrize("p", [0.25, 0.5])
def test_partial_fraction_corrupts_a_fixed_subset_only(p):
    n = 237
    a, b = ShuffledMaskDataset(_Ids(n), 5, p), ShuffledMaskDataset(_Ids(n), 5, p)
    assert np.array_equal(a.perm, b.perm)                                  # seeded
    k = int(round(p * n))
    moved = a.perm != np.arange(n)
    assert moved.sum() == k == len(a.corrupted)                            # every subset item gets another mask
    assert set(a.perm[moved]) == set(np.where(moved)[0])                   # masks come from inside the subset
    assert all(a[i] == (i, i) for i in np.where(~moved)[0])                # the rest keep their own masks


def test_too_small_fraction_raises():
    with pytest.raises(ValueError):
        ShuffledMaskDataset(_Ids(10), 0, 0.1)
