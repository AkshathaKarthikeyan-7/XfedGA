"""HD95 conventions and agreement with the legacy implementation; config validation."""
import math

import pytest
import torch

from fedga.evaluation import hd95_norm, hd95_px
from fedga.ga_optimizer import hd95 as legacy_hd95


def _square(size, top, left, side):
    m = torch.zeros(size, size)
    m[top:top + side, left:left + side] = 1
    return m


@pytest.mark.parametrize("shift", [0, 3, 20, 120])
def test_hd95_matches_legacy(shift):
    a, b = _square(384, 50, 50, 40), _square(384, 50 + shift, 50, 40)
    assert hd95_norm(a, b, 384.0) == pytest.approx(legacy_hd95(a, b, 384.0))
    assert hd95_px(a, b) == pytest.approx(min(legacy_hd95(a, b, 384.0), 1.0) * 384.0)


def test_hd95_empty_mask_conventions():
    empty, square = torch.zeros(384, 384), _square(384, 10, 10, 20)
    assert hd95_norm(empty, empty, 384.0) == 0.0 == legacy_hd95(empty, empty)
    assert hd95_norm(square, empty, 384.0) == 1.0 == legacy_hd95(square, empty)
    assert hd95_px(square, empty) is None


def test_identical_masks_have_zero_hd95():
    m = _square(384, 100, 100, 30)
    assert hd95_px(m, m) == 0.0


def test_hd95_clipped_to_one():
    a, b = _square(384, 0, 0, 2), _square(384, 380, 380, 2)
    assert hd95_px(a, b) > 384 and hd95_norm(a, b, 384.0) == 1.0
    assert math.isclose(legacy_hd95(a, b, 384.0), 1.0)


def test_config_validation_rejects_bad_settings():
    from experiments.config import load_config
    cfg = load_config("breast")
    assert cfg["fitness"]["weights"] == [0.5, 0.3, 0.2] and cfg["aggregation"]["beta"] == 0.5
    with pytest.raises(ValueError):
        load_config("breast", ["fitness.weights=[0.6, 0.3, 0.2]"])
    with pytest.raises(ValueError):
        load_config("breast", ["fitness.weights=[1.2, -0.2, 0.0]"])
    with pytest.raises(ValueError):
        load_config("prostate", ["fitness.hd95_normalizer=543"])
    with pytest.raises(KeyError):
        load_config("prostate", ["ga.populaton=4"])
    assert load_config("prostate", ["aggregation.beta=1.0"])["aggregation"]["beta"] == 1.0
