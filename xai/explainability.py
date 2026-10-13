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


def get_gradcam(model, target_layer, input_tensor, target_class=1):
    """
    target_layer: the encoder layer to hook, e.g. the last conv block before
    the bottleneck for a MONAI UNet — inspect `model` and pick a `nn.Conv2d`
    or residual block near the deepest encoder stage.
    Returns a (H, W) numpy saliency map normalized to [0, 1].
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

    model.eval()
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        targets = [SemanticSegmentationTarget(target_class, np.ones(input_tensor.shape[-2:]))]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam  # already normalized [0,1] by the library


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


def insertion_deletion_auc(model, input_tensor, saliency_map, target_class=1,
                            steps=20, mode="deletion"):
    """
    Deletion: progressively zero out the most-salient pixels (per saliency_map,
    highest first) and track how fast the predicted DSC/confidence for
    target_class collapses -- a faithful explanation should collapse fast
    (low AUC). Insertion is the reverse (start blank, add back top pixels,
    good explanation -> confidence rises fast, high AUC).

    Returns the AUC of the confidence-vs-fraction-removed(or added) curve.
    """
    model.eval()
    flat_order = np.argsort(-saliency_map.flatten())  # most salient first
    h, w = saliency_map.shape
    total_pixels = h * w
    chunk = total_pixels // steps

    img = input_tensor.clone()
    scores = []
    if mode == "insertion":
        working = torch.zeros_like(img)
    else:
        working = img.clone()

    for step in range(steps + 1):
        idx = flat_order[: step * chunk]
        mask = torch.zeros(total_pixels, dtype=torch.bool)
        mask[idx] = True
        mask = mask.view(h, w)

        current = working.clone()
        if mode == "insertion":
            current[..., mask] = img[..., mask]
        else:
            current[..., mask] = 0.0

        with torch.no_grad():
            out = model(current)
            prob = F.softmax(out, dim=1)[0, target_class].mean().item()
        scores.append(prob)

    return float(np.trapz(scores, dx=1.0 / steps))
