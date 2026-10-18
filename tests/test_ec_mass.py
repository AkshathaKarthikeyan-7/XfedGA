"""EC_mass (share of CAM mass inside the mask; XFedGA+ EC, pre-registered 18 Oct 2026)."""
import numpy as np
import pytest

from xai.explainability import cam_mass_in_mask, explanation_consistency


def test_ec_mass_basic_values():
    m = np.zeros((10, 10))
    m[2:5, 2:5] = 1
    inside = np.zeros((10, 10))
    inside[3, 3] = 1.0
    assert cam_mass_in_mask(inside, m) == 1.0
    assert cam_mass_in_mask(np.ones((10, 10)), m) == pytest.approx(9 / 100)
    assert cam_mass_in_mask(np.zeros((10, 10)), m) == 0.0


def test_ec_mass_ignores_peak_sharpness_unlike_iou():
    """A sharp peak inside a large object: IoU@0.5 is tiny, EC_mass is 1 (the v2-crop situation)."""
    m = np.zeros((100, 100))
    m[20:80, 20:80] = 1                                   # large object (36 % of the image)
    yy, xx = np.mgrid[:100, :100]
    peak = np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / (2 * 2.0 ** 2))
    assert explanation_consistency(peak, m, 0.5) < 0.02
    assert cam_mass_in_mask(peak, m) > 0.99


@pytest.mark.slow
def test_ec_mass_known_answer(trained):
    """Synthetic known-answer model (tests/test_xai_known_answer.py): EC_mass of its full-res Grad-CAM is high."""
    from tests.test_xai_known_answer import _cam
    model, x, y, _ = trained
    vals = [cam_mass_in_mask(_cam(model, x[i:i + 1], "pred", "fullres"), y[i]) for i in range(len(y))]
    assert min(vals) > 0.5 and np.mean(vals) > 0.9


@pytest.fixture(scope="module")
def trained():
    from tests.test_xai_known_answer import trained as fx
    return fx.__wrapped__() if hasattr(fx, "__wrapped__") else fx()
