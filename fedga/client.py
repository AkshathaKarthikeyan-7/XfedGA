"""
Client-side core functions, shared by the custom loop (experiments/run_federated.py)
and the Flower clients (fl_dist/, Session 3). Framework-agnostic: plain PyTorch.

Assumption (spec section 4): q_i is computed CLIENT-SIDE, on the client's own
fixed q_i subsample, from the client's locally trained model; no image or
mask leaves the client, only (w_i, n_i, q_i) are sent to the server.
"""
import logging

import numpy as np
import torch

from xai.explainability import explanation_consistency, get_gradcam, unet_cam_layer

log = logging.getLogger(__name__)


def local_update(model, loader, epochs, lr, betas, device, loss_fn):
    """E local epochs of Adam on the client's training loader; returns the new state_dict."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=tuple(betas))
    for _ in range(epochs):
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)
            opt.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def client_saliency_score(model, loader, device, cam_threshold=0.5):
    """
    q_i = mean over the client's fixed q_i subsample of
          EC(CAM_k, M_k) = |{CAM_k >= t} ∩ M_k| / |{CAM_k >= t} ∪ M_k|   (t = cam_threshold),
    where CAM_k is the Grad-CAM of the summed class-1 logit at unet_cam_layer, min-max
    normalised per image. A Grad-CAM failure on an image counts as EC = 0 and is logged;
    an empty loader returns 0.5 (neutral) and is logged.
    """
    model.eval()
    layer = unet_cam_layer(model)
    ecs, failures = [], 0
    for imgs, masks in loader:
        for i in range(imgs.shape[0]):
            try:
                cam = get_gradcam(model, layer, imgs[i:i + 1].to(device), target_class=1)
                ecs.append(explanation_consistency(cam, masks[i].numpy(), threshold=cam_threshold))
            except Exception:  # noqa: BLE001 -- counted and logged, matches the pre-refactor behaviour
                failures += 1
                ecs.append(0.0)
    if failures:
        log.warning("q_i: Grad-CAM failed on %d/%d images (counted as EC=0)", failures, len(ecs))
    if not ecs:
        log.warning("q_i: empty q_i subsample, returning 0.5")
        return 0.5
    return float(np.mean(ecs))
