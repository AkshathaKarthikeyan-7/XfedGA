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


def test_xai_functions_on_a_random_model():
    """Grad-CAM++, deletion/insertion AUC and the randomisation check run and return sane values.
    pytorch-grad-cam places its target mask on CUDA whenever CUDA exists, so the model follows."""
    from models.residual_unet import build_residual_unet
    from xai.explainability import get_gradcam, get_gradcam_pp, insertion_deletion_auc, randomization_check, unet_cam_layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_residual_unet().to(device).eval()
    x = torch.rand(1, 1, 64, 64, device=device)
    cam = get_gradcam(model, unet_cam_layer(model), x)
    campp = get_gradcam_pp(model, unet_cam_layer(model), x)
    assert cam.shape == campp.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1
    for mode in ("deletion", "insertion"):
        auc = insertion_deletion_auc(model, x, cam, steps=4, mode=mode)
        assert 0.0 <= auc <= 1.0
    rho = randomization_check(model, x, steps=(1, 4), size=16)
    assert set(rho) == {1, 4} and all(-1.0 <= v <= 1.0 for v in rho.values())
