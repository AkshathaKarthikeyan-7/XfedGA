"""
Known-answer test for the explanation metrics: a Residual U-Net briefly trained on synthetic images
(dark noise + one bright square = the lesion) must, on a new image with the square at a known place,
(1) segment the square; (2) concentrate its Grad-CAM in the square and hit it in the pointing game;
(3) give insertion AUC > deletion AUC; (4) lose rank correlation when its weights are randomised.
FINDING pinned here (strict xfail): the layer used by every run up to 17 Oct 2026 (unet_cam_layer, a
half-resolution decoder conv) fails (2) with either CAM target; the full-resolution decoder up-conv
(unet_fullres_cam_layer) passes (2)-(4). Slow (a few seconds of GPU training).
"""
import numpy as np
import pytest
import torch

from fedga.losses import build_loss
from models.residual_unet import build_residual_unet
from xai.explainability import get_gradcam, insertion_deletion_auc, pointing_game_accuracy, randomization_check, \
    unet_cam_layer, unet_fullres_cam_layer

LAYERS = {"current": unet_cam_layer, "fullres": unet_fullres_cam_layer}

SIZE = 64


def sample(rng, n):
    x = rng.normal(0.2, 0.05, size=(n, 1, SIZE, SIZE)).astype(np.float32)
    y = np.zeros((n, SIZE, SIZE), dtype=np.int64)
    for i in range(n):
        s = rng.integers(10, 18)
        r, c = rng.integers(4, SIZE - s - 4, size=2)
        x[i, 0, r:r + s, c:c + s] += 0.6
        y[i, r:r + s, c:c + s] = 1
    return torch.from_numpy(np.clip(x, 0, 1)), torch.from_numpy(y)


@pytest.fixture(scope="module")
def trained():
    from experiments.config import seed_everything
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    model = build_residual_unet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = build_loss()
    for _ in range(250):
        x, y = sample(rng, 8)
        opt.zero_grad()
        loss_fn(model(x.to(device)), y.unsqueeze(1).to(device)).backward()
        opt.step()
    model.eval()
    x, y = sample(np.random.default_rng(123), 5)
    return model, x.to(device), y.numpy(), device


def _cam(model, x, cam_target, layer="current"):
    return get_gradcam(model, LAYERS[layer](model), x, cam_target=cam_target)


@pytest.mark.slow
def test_model_segments_the_square(trained):
    model, x, y, _ = trained
    with torch.no_grad():
        pred = torch.argmax(model(x), 1).cpu().numpy()
    dice = [2 * (p * t).sum() / (p.sum() + t.sum()) for p, t in zip(pred, y)]
    assert min(dice) > 0.8


def _localisation(model, x, y, cam_target, layer):
    mass, pg = [], []
    for i in range(len(y)):
        cam = _cam(model, x[i:i + 1], cam_target, layer)
        mass.append(cam[y[i] > 0].sum() / max(cam.sum(), 1e-9))
        pg.append(pointing_game_accuracy(cam, y[i]))
    return np.array(mass), np.array(pg)


@pytest.mark.slow
@pytest.mark.parametrize("cam_target", ["pred", "all"])
def test_fullres_layer_localises_the_square(trained, cam_target):
    model, x, y, _ = trained
    mass, pg = _localisation(model, x, y, cam_target, "fullres")
    assert mass.min() > 0.5 and pg.mean() == 1.0          # square covers ~3-8 % of the image


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="FINDING 17 Oct 2026: the layer used by every run so far (unet_cam_layer, "
                                       "half-resolution decoder conv) does not localise the lesion on a near-perfect "
                                       "model; strict, so a change in this behaviour is noticed")
@pytest.mark.parametrize("cam_target", ["pred", "all"])
def test_current_layer_localises_the_square(trained, cam_target):
    model, x, y, _ = trained
    mass, pg = _localisation(model, x, y, cam_target, "current")
    assert mass.min() > 0.5 and pg.mean() >= 0.8


@pytest.mark.slow
@pytest.mark.parametrize("cam_target", ["pred", "all"])
def test_curves_start_and_end_at_the_original_image(trained, cam_target):
    model, x, y, _ = trained
    cam = _cam(model, x[:1], cam_target, "fullres")
    _, ins_curve = insertion_deletion_auc(model, x[:1], cam, gt_mask=y[0], mode="insertion", return_curve=True)
    _, del_curve = insertion_deletion_auc(model, x[:1], cam, gt_mask=y[0], mode="deletion", return_curve=True)
    assert del_curve[0] == pytest.approx(ins_curve[-1], abs=1e-6)      # both = the untouched image
    assert ins_curve[0] == pytest.approx(del_curve[-1], abs=1e-6)      # both = the fully blurred image


@pytest.mark.slow
def test_insertion_beats_deletion_with_a_faithful_map(trained):
    """With the localising (full-res) CAM, restoring the salient pixels must help the model more than
    removing them hurts it: insertion AUC > deletion AUC on every image."""
    model, x, y, _ = trained
    for i in range(len(y)):
        cam = _cam(model, x[i:i + 1], "pred", "fullres")
        ins = insertion_deletion_auc(model, x[i:i + 1], cam, gt_mask=y[i], mode="insertion")
        dele = insertion_deletion_auc(model, x[i:i + 1], cam, gt_mask=y[i], mode="deletion")
        assert ins > dele, (i, ins, dele)


@pytest.mark.slow
def test_randomisation_lowers_rank_correlation(trained):
    model, x, _, _ = trained
    rho = randomization_check(model, x[:1], steps=(1, 16), size=32, cam_target="pred", layer_fn=unet_fullres_cam_layer)
    assert rho[16] < 0.5 and rho[16] <= rho[1]


@pytest.mark.slow
def test_full_randomisation_step(trained):
    model, x, _, _ = trained
    rho = randomization_check(model, x[:1], steps=(1, "all"), size=32, cam_target="pred", layer_fn=unet_fullres_cam_layer)
    assert set(rho) == {1, "all"} and rho["all"] < 0.5
