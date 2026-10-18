"""
Explainability utilities:
  - Grad-CAM wrapper for the Residual U-Net (works for any conv encoder layer)
  - Attention-map extraction for the Attention U-Net (free explanations, no
    extra library needed — just hook the attention gate's output)
  - Faithfulness metrics: pointing-game accuracy, insertion/deletion AUC

Uses `pytorch-grad-cam` (pip install grad-cam) for the CAM implementation so
you're not reinventing gradient hooking from scratch, but keep this module
thin so you can swap CAM variants (Grad-CAM++, Score-CAM) with one line.
"""
import numpy as np
import torch
import torch.nn.functional as F


def explanation_consistency(saliency_map, ground_truth_mask, threshold=0.5):
    """
    EC(M): IoU between thresholded saliency map and ground-truth mask.
    saliency_map: (H, W) float array normalized to [0,1] (from Grad-CAM etc.)
    ground_truth_mask: (H, W) binary array
    """
    sal_bin = (saliency_map >= threshold).astype(np.float32)
    gt = ground_truth_mask.astype(np.float32)
    inter = (sal_bin * gt).sum()
    union = ((sal_bin + gt) > 0).sum()
    return float(inter / union) if union > 0 else 0.0


CAM_TARGETS = ("all", "pred")


def cam_target_mask(model, input_tensor, target_class=1, cam_target="all"):
    """
    Pixels whose class-`target_class` logits are summed to form the Grad-CAM target score:
      "all"  -- every pixel (the definition used by all runs up to 17 Oct 2026);
      "pred" -- the model's predicted `target_class` pixels (argmax); falls back to "all" if the
                prediction is empty. Uses no ground truth.
    """
    if cam_target not in CAM_TARGETS:
        raise ValueError(f"cam_target must be one of {CAM_TARGETS}, got {cam_target!r}")
    h, w = input_tensor.shape[-2:]
    if cam_target == "pred":
        with torch.no_grad():
            pred = (torch.argmax(model(input_tensor), dim=1)[0] == target_class).float().cpu().numpy()
        if pred.sum() > 0:
            return pred
    return np.ones((h, w), dtype=np.float32)


def get_gradcam(model, target_layer, input_tensor, target_class=1, cam_target="pred"):
    """
    Grad-CAM of the summed class-`target_class` logits over the pixels chosen by `cam_target`
    (see cam_target_mask), at `target_layer` (for the Residual U-Net: unet_cam_layer, the last decoder
    conv, 16 channels, half resolution). pytorch-grad-cam upsamples the map to the input size and
    min-max normalises it per image, so the map is (H, W) in [0, 1] and thresholds are relative.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    model.eval()
    tmask = cam_target_mask(model, input_tensor, target_class, cam_target)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = [SemanticSegmentationTarget(target_class, tmask)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam


def get_attention_map(attention_unet_output_hook):
    """
    For AttentionUnet (MONAI), register a forward hook on the attention gate
    module(s) and capture the attention coefficients (alpha) directly —
    these ARE the model's learned "where am I looking" map, no gradient
    computation needed. Example hook setup:

        activations = {}
        def hook(module, input, output):
            activations['attn'] = output.detach()
        model.some_attention_block.register_forward_hook(hook)

    This function is a placeholder to remind you to wire that hook up per
    the exact MONAI AttentionUnet internals (check `model` structure first
    with `print(model)` once you load it, attention blocks are named
    `up_kernel_size` submodules or similar — verify against your installed
    MONAI version).
    """
    raise NotImplementedError(
        "Register a forward hook on the AttentionUnet's attention-gate module; "
        "see docstring for the pattern once you've inspected the model structure."
    )


def pointing_game_accuracy(saliency_map, ground_truth_mask):
    """
    Does the single highest-saliency pixel fall inside the ground-truth mask?
    Standard weak-localization faithfulness metric — report as a % over the
    test set.
    """
    peak_idx = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return float(ground_truth_mask[peak_idx] > 0)


def _blurred(input_tensor, sigma):
    import cv2
    x = input_tensor.detach().cpu().numpy()
    out = np.stack([[cv2.GaussianBlur(ch, (0, 0), sigma) for ch in img] for img in x])
    return torch.from_numpy(out).to(input_tensor.device, input_tensor.dtype)


def insertion_deletion_auc(model, input_tensor, saliency_map, gt_mask=None, target_class=1, steps=20,
                           mode="deletion", baseline="blur", blur_sigma=8.0, return_curve=False):
    """
    Deletion / insertion curves (Petsiuk et al., 2018) adapted to segmentation.
      measured output: the mean predicted probability of `target_class` INSIDE the ground-truth mask
                       `gt_mask` (H, W) -- the model's confidence on the true lesion/organ. Without
                       gt_mask: the mean over the whole image (legacy; dominated by background).
      order:           pixels sorted by saliency, most salient first; `steps` equal chunks, the
                       last step covering every pixel (earlier code left total % steps pixels out).
      deletion:        the top-k salient pixels are replaced by the baseline image; a faithful map
                       makes the score fall fast (LOW AUC is better).
      insertion:       start from the baseline image and restore the top-k salient pixels from the
                       original; a faithful map makes the score rise fast (HIGH AUC is better).
      baseline:        "blur" = Gaussian blur of the image (sigma = blur_sigma px, removes local
                       structure but keeps intensity statistics); "zero" = black (legacy).
    AUC = trapezoid over the fraction of pixels removed / inserted (0..1).
    """
    model.eval()
    flat_order = np.argsort(-saliency_map.flatten(), kind="stable")
    h, w = saliency_map.shape
    total = h * w
    counts = [int(round(step * total / steps)) for step in range(steps + 1)]   # 0 .. total, every pixel by the end
    img = input_tensor.clone()
    base = _blurred(img, blur_sigma) if baseline == "blur" else torch.zeros_like(img)
    region = torch.from_numpy(np.asarray(gt_mask) > 0).to(img.device) if gt_mask is not None else None
    if region is not None and region.sum() == 0:
        region = None
    scores = []
    for step in range(steps + 1):
        sel = torch.zeros(total, dtype=torch.bool)
        sel[flat_order[: counts[step]]] = True
        sel = sel.view(h, w).to(img.device)
        if mode == "insertion":
            current = torch.where(sel, img, base)
        elif mode == "deletion":
            current = torch.where(sel, base, img)
        else:
            raise ValueError(mode)
        with torch.no_grad():
            prob = F.softmax(model(current), dim=1)[0, target_class]
        scores.append(float(prob[region].mean() if region is not None else prob.mean()))
    auc = float(np.trapz(scores, dx=1.0 / steps))
    return (auc, scores) if return_curve else auc


def unet_cam_layer(model):
    """Grad-CAM target layer for the MONAI Residual U-Net (models/residual_unet.py) used by every run up to
    17 Oct 2026: a conv inside the last half-resolution decoder residual unit (16 channels).
    Known-answer test (tests/test_xai_known_answer.py): on a near-perfect synthetic model this layer does
    NOT localise the lesion (CAM mass in lesion 0.05-0.19, pointing game 0.38) -- see unet_fullres_cam_layer."""
    return model.model[1].submodule[2][1].conv.unit0.conv


def unet_fullres_cam_layer(model):
    """XFedGA's Grad-CAM layer from 17 Oct 2026 (decision (b), plan.md section 0.0): the full-resolution decoder
    up-convolution (model.model[2][0]), just before the output head. Passes the known-answer test (CAM mass in
    lesion 1.00, pointing game 1.00, tests/test_xai_known_answer.py)."""
    return model.model[2][0]


CAM_LAYERS = {"fullres": unet_fullres_cam_layer, "halfres": unet_cam_layer}


def cam_layer(model, name="fullres"):
    """The Grad-CAM layer named in the config (fitness.cam_layer): fullres (method default) | halfres (legacy)."""
    if name not in CAM_LAYERS:
        raise ValueError(f"cam_layer must be one of {sorted(CAM_LAYERS)}, got {name!r}")
    return CAM_LAYERS[name](model)


def get_gradcam_pp(model, target_layer, input_tensor, target_class=1, cam_target="pred"):
    """Grad-CAM++ (a second CAM method for the independent explanation metrics), same target as get_gradcam."""
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    model.eval()
    tmask = cam_target_mask(model, input_tensor, target_class, cam_target)
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        return cam(input_tensor=input_tensor, targets=[SemanticSegmentationTarget(target_class, tmask)])[0]


def randomization_check(model, input_tensor, steps=(1, 2, 4, 8, 16, "all"), seed=0, size=96, cam_target="pred",
                        layer_fn=None):
    """
    Cascading model-parameter randomisation (Adebayo et al., 2018): re-initialise the conv layers from the
    output towards the input (cumulatively, `steps` = number of layers randomised) and return the Spearman
    rank correlation between the original Grad-CAM and the Grad-CAM of each randomised model (maps
    downsampled to size x size). A faithful explanation should decorrelate as more layers are randomised.
    The step "all" re-initialises every conv layer ("full parameter randomisation", pre-registered H-XAI-2).
    """
    import copy

    import torch.nn as nn
    from scipy.stats import spearmanr

    def cam_of(m):
        c = get_gradcam(m, (layer_fn or unet_fullres_cam_layer)(m), input_tensor, target_class=1, cam_target=cam_target)
        return F.interpolate(torch.from_numpy(c)[None, None].float(), size=(size, size), mode="area").flatten().numpy()

    base = cam_of(model)
    rand = copy.deepcopy(model)
    convs = [m for m in rand.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))][::-1]
    out, done = {}, 0
    for step in steps:
        k = len(convs) if step == "all" else min(step, len(convs))
        for i in range(done, k):
            torch.manual_seed(seed + i)
            convs[i].reset_parameters()
        done = k
        c = cam_of(rand)
        rho = spearmanr(base, c).correlation if np.std(c) > 0 and np.std(base) > 0 else 0.0
        out[step] = float(rho)
    return out
