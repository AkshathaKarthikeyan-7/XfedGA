"""
Server-side evaluation: per-image metrics and the GA fitness functions.
Framework-agnostic; shared by the custom loop and fl_dist/.

Assumption (decision Q4, 15 Oct 2026): every image and mask is resized to
image_size x image_size (384 x 384) by the data loaders BEFORE any metric is
computed, so HD95 is measured in pixels of that grid and HD95_norm =
min(HD95_px / hd95_normalizer, 1) with hd95_normalizer = 384 (the image side).
Conventions: both masks empty -> HD95_norm 0; exactly one empty -> 1.
"""
import copy

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric

from fedga.ga_optimizer import dice_score, multi_objective_fitness
from xai.explainability import cam_layer as get_cam_layer, cam_mass_in_mask, explanation_consistency, get_gradcam, \
    pointing_game_accuracy

_HD95 = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")


def hd95_px(pred, target):
    """95th-percentile Hausdorff distance in pixels; None if either mask is empty."""
    if pred.sum() == 0 or target.sum() == 0:
        return None
    _HD95.reset()
    _HD95(y_pred=pred[None, None].float(), y=target[None, None].float())
    return float(_HD95.aggregate().item())


def hd95_norm(pred, target, normalizer):
    """HD95 in pixels divided by `normalizer` (the image side), capped at 1; 0 if both masks are empty, 1 if one is."""
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    raw = hd95_px(pred, target)
    return 1.0 if raw is None else min(raw / normalizer, 1.0)


class FixedSet:
    """A small fixed image set held on the device (the GA fitness subsample)."""

    def __init__(self, dataset, indices, device):
        items = [dataset[i] for i in indices]
        self.images = torch.stack([x for x, _ in items]).to(device)
        self.masks = torch.stack([y for _, y in items])  # CPU, (N, H, W) long
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)


@torch.no_grad()
def _predict(model, images, batch_size=8):
    return torch.cat([torch.argmax(model(images[i:i + batch_size]), dim=1)
                      for i in range(0, len(images), batch_size)])


def per_image_metrics(model, batches, device, hd95_normalizer, cam_threshold=0.5, with_cam=True, cam_target="pred",
                      cam_layer="fullres"):
    """
    batches: iterable of (images (B,1,H,W), masks (B,H,W)). Returns one dict per image:
    dsc, hd95_norm, hd95_px (NaN if a mask is empty), and if with_cam: ec, pg.
    """
    model.eval()
    layer = get_cam_layer(model, cam_layer) if with_cam else None
    rows = []
    for imgs, masks in batches:
        preds = _predict(model, imgs.to(device)).cpu()
        for k in range(len(preds)):
            p, m = preds[k].float(), masks[k].float()
            raw = hd95_px(p, m)
            row = {"dsc": float(dice_score(p, m)), "hd95_norm": hd95_norm(p, m, hd95_normalizer),
                   "hd95_px": float("nan") if raw is None else raw}
            if with_cam:
                try:
                    cam = get_gradcam(model, layer, imgs[k:k + 1].to(device), target_class=1, cam_target=cam_target)
                    row["ec"] = explanation_consistency(cam, masks[k].numpy(), threshold=cam_threshold)
                    row["pg"] = pointing_game_accuracy(cam, masks[k].numpy())
                    row["ec_mass"] = cam_mass_in_mask(cam, masks[k].numpy())
                except Exception:  # noqa: BLE001 -- recorded as NaN, never silently dropped
                    row["ec"], row["pg"], row["ec_mass"] = float("nan"), float("nan"), float("nan")
            rows.append(row)
    return rows


def fitness_dsc(model, fixed, record=None):
    """FedGA (baseline) fitness: mean DSC over the fixed fitness subsample."""
    model.eval()
    preds = _predict(model, fixed.images).cpu()
    f = float(np.mean([float(dice_score(preds[k].float(), fixed.masks[k].float())) for k in range(len(fixed))]))
    if record is not None:
        record.append({"dsc": f, "fitness": f})
    return f


def fitness_multi(model, fixed, weights, hd95_normalizer, cam_threshold, record=None, cam_target="pred",
                  cam_layer="fullres", ec_kind="iou"):
    """XFedGA fitness: w1*mean DSC - w2*mean HD95_norm + w3*mean EC over the fixed subsample.
    ec_kind: "iou" (EC = IoU of the CAM thresholded at cam_threshold; XFedGA) | "mass" (EC = share of CAM mass
    inside the mask; XFedGA+). With `record`, appends the components so the candidate pool can be re-ranked offline
    under other fitness weights (fitness-weight Step 0); with ec_kind "mass" both EC values are recorded."""
    if ec_kind not in ("iou", "mass"):
        raise ValueError(f"ec_kind must be iou|mass, got {ec_kind!r}")
    rows = per_image_metrics(model, [(fixed.images[i:i + 1], fixed.masks[i:i + 1]) for i in range(len(fixed))],
                             fixed.images.device, hd95_normalizer, cam_threshold, with_cam=True, cam_target=cam_target,
                             cam_layer=cam_layer)
    key = "ec" if ec_kind == "iou" else "ec_mass"
    ec = [0.0 if np.isnan(r[key]) else r[key] for r in rows]
    dsc, hd, e = float(np.mean([r["dsc"] for r in rows])), float(np.mean([r["hd95_norm"] for r in rows])), float(np.mean(ec))
    f = multi_objective_fitness(dsc, hd, e, weights=tuple(weights))
    if record is not None:
        rec = {"dsc": dsc, "hd95_norm": hd, "ec": e, "fitness": f}
        if ec_kind == "mass":
            rec["ec_iou"] = float(np.mean([0.0 if np.isnan(r["ec"]) else r["ec"] for r in rows]))
        record.append(rec)
    return f


def build_fitness_fn(model_template, fixed, kind, weights=(0.5, 0.3, 0.2), hd95_normalizer=384.0, cam_threshold=0.5,
                     record=None, cam_target="pred", cam_layer="fullres", ec_kind="iou"):
    """state_dict -> fitness, reusing one evaluation model. kind: 'dsc' or 'multi'.
    record: optional list; every evaluation appends its components (in evaluation order)."""
    model = copy.deepcopy(model_template).to(fixed.images.device)

    def evaluate(state_dict):
        model.load_state_dict(state_dict)
        if kind == "dsc":
            return fitness_dsc(model, fixed, record)
        if kind == "multi":
            return fitness_multi(model, fixed, weights, hd95_normalizer, cam_threshold, record, cam_target, cam_layer,
                                 ec_kind)
        raise ValueError(f"unknown fitness kind {kind!r}")
    return evaluate


def server_client_scores(model_template, fixed, client_states, kind, hd95_normalizer=384.0, cam_threshold=0.5,
                         cam_target="pred", cam_layer="fullres", ec_kind="iou"):
    """
    XFedGA+ server-verified client score: each client model is evaluated by the SERVER on the fixed fitness
    subsample (server data; no validation or test data). kind "dsc_ec": s_i = DSC_i * EC_i; kind "dsc": s_i = DSC_i
    (control). Both kinds compute the same components, so the cost is identical. Returns one dict per client.
    """
    if kind not in ("dsc_ec", "dsc"):
        raise ValueError(f"unknown server score kind {kind!r}")
    rec = []
    fn = build_fitness_fn(model_template, fixed, "multi", (1.0, 0.0, 0.0), hd95_normalizer, cam_threshold, rec,
                          cam_target, cam_layer, ec_kind)
    out = []
    for st in client_states:
        fn(st)
        r = rec[-1]
        out.append({"dsc": r["dsc"], "ec": r["ec"], "s": r["dsc"] * r["ec"] if kind == "dsc_ec" else r["dsc"]})
    return out
