"""
Protocol v2 options (plan.md, pre-registered 18 Oct 2026): prostate FOV crop + z-score preprocessing,
fitness subsample capped at the fitness-set size, and logging of w_agg's validation DSC.
The v1 defaults must leave every v1 result unchanged (the regression pin in test_regression.py covers that).
"""
import json

import numpy as np
import pytest

from experiments.config import load_config
from tests.conftest import require_path


def test_center_crop_is_image_centred_and_pads():
    from data.prostate_dataset import _center_crop
    a = np.arange(10 * 12).reshape(10, 12)
    c = _center_crop(a, 4)
    assert c.shape == (4, 4) and np.array_equal(c, a[3:7, 4:8])
    p = _center_crop(np.ones((3, 5)), 7)                  # smaller than the crop: zero-padded, content centred
    assert p.shape == (7, 7) and p.sum() == 15 and p[2:5, 1:6].all()


def test_zscore_uses_the_whole_cropped_volume_and_no_masks():
    from data.prostate_dataset import _crop_normalize_resize
    rng = np.random.RandomState(0)
    imgs = [rng.rand(300, 300).astype(np.float32) * 100 for _ in range(4)]
    m1 = [np.zeros((300, 300)) for _ in range(4)]
    for m in m1[1:3]:
        m[140:160, 140:160] = 1
    m2 = [m.copy() for m in m1]
    m2[1][145:150, 145:150] = 0                             # different mask, same images
    a = _crop_normalize_resize(list(zip(imgs, m1)), (64, 64), 100, "zscore", 0.0)
    b = _crop_normalize_resize(list(zip(imgs, m2)), (64, 64), 100, "zscore", 0.0)
    assert len(a) == len(b) == 2                            # empty slices dropped after the statistics
    for (xa, _), (xb, _) in zip(a, b):
        assert np.array_equal(xa, xb)                       # intensities do not depend on the masks
    vol = np.stack([i[50:250, 50:250] for i in imgs]).astype(np.float64)
    import cv2
    ref = (cv2.resize(imgs[1][50:250, 50:250], (64, 64), interpolation=cv2.INTER_LINEAR) - vol.mean()) / vol.std()
    assert np.allclose(a[0][0], ref, atol=1e-5)


def test_fitness_subsample_capped_at_set_size():
    from data.splits import evenly_spaced_per_group
    assert evenly_spaced_per_group([15, 12, 20], 47) == list(range(47))
    assert evenly_spaced_per_group([15, 12, 20], 120) == list(range(47))
    assert len(evenly_spaced_per_group([15, 12, 20], 30)) == 30


def test_preprocessing_config_validation():
    assert load_config("prostate")["preprocessing"] == {"fov_crop_mm": None, "intensity": "minmax"}
    load_config("prostate", ["preprocessing.fov_crop_mm=100", "preprocessing.intensity=zscore"])
    with pytest.raises(ValueError):
        load_config("prostate", ["preprocessing.intensity=robust"])
    with pytest.raises(ValueError):
        load_config("prostate", ["preprocessing.fov_crop_mm=0"])
    assert "preprocessing" not in load_config("breast")


@pytest.mark.data
def test_v1_cache_key_unchanged_and_v2_key_differs():
    require_path("raw data/PROMISE12")
    from data import prostate_dataset as pd
    seen = {}
    real = pd.build_or_load

    def spy(name, spec, *a, **k):
        seen.setdefault(name, []).append(spec)
        raise StopIteration                      # stop before building anything
    pd.build_or_load = spy
    try:
        for kw in ({}, {"fov_crop_mm": 100, "intensity": "zscore"}):
            with pytest.raises(StopIteration):
                pd.prostate_cache("PROMISE12", "raw data/PROMISE12", **kw)
    finally:
        pd.build_or_load = real
    v1, v2 = seen["PROMISE12"]
    assert "fov_crop_mm" not in v1 and "intensity" not in v1 and "code_v2" not in v1
    assert {k: v for k, v in v2.items() if k not in ("fov_crop_mm", "intensity", "code_v2")} == v1


@pytest.mark.data
@pytest.mark.slow
def test_v2_options_run_and_wagg_logging_does_not_change_the_run(tmp_path):
    require_path("raw data/PROMISE12")
    from tests.test_switches import _run
    base = _run(tmp_path / "a", "xfedga")["rounds"][0]
    logged = _run(tmp_path / "b", "xfedga", ["evaluation.log_wagg_val=true",
                                             "evaluation.log_client_val=true"])["rounds"][0]
    assert "val_dsc_wagg" not in base and 0.0 <= logged["val_dsc_wagg"] <= 1.0
    assert len(json.loads(logged["client_val_dsc_i"])) == 2
    for k in ("val_dsc", "q_i", "alpha_i", "ga_best_per_gen"):     # evaluation only: same RNG stream, same result
        assert base[k] == logged[k], k
    v2 = _run(tmp_path / "c", "fedga", ["preprocessing.fov_crop_mm=100", "preprocessing.intensity=zscore",
                                       "evaluation.log_wagg_val=true"])["rounds"][0]
    assert 0.0 <= v2["val_dsc"] <= 1.0 and 0.0 <= v2["val_dsc_wagg"] <= 1.0
    assert json.loads(v2["alpha_i"])
