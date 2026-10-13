"""
Quick sanity check: does the Grad-CAM target layer produce a sensible
feature map for prostate images, and does the resulting heatmap land
anywhere near the actual prostate region?
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.residual_unet import build_residual_unet
from data.prostate_dataset import ProstateClientDataset, _index_decathlon
from xai.explainability import get_gradcam

def _gradcam_target_layer(model):
    return model.model[1].submodule[2][1].conv.unit0.conv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_residual_unet().to(device)
model.load_state_dict(torch.load("global_model_base_fedga_prostate_BEST.pt", map_location=device))
model.eval()

target_layer = _gradcam_target_layer(model)

# Print feature map shape at the target layer for a real prostate input
activation = {}
def hook(module, inp, out):
    activation["shape"] = out.shape
handle = target_layer.register_forward_hook(hook)

decathlon_root = "raw data/Task05_Prostate/Task05_Prostate"
cases = _index_decathlon(decathlon_root)
ds = ProstateClientDataset("Decathlon", decathlon_root, cases=cases[:3])

img, mask = ds[0]
with torch.no_grad():
    model(img.unsqueeze(0).to(device))
handle.remove()
print(f"Target layer feature map shape for prostate input: {activation['shape']}")

# Visualize Grad-CAM on 3 real slices with nonzero prostate mask
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
shown = 0
for i in range(len(ds)):
    img, mask = ds[i]
    if mask.sum() == 0:
        continue
    img_in = img.unsqueeze(0).to(device)
    cam = get_gradcam(model, target_layer, img_in, target_class=1)

    axes[shown][0].imshow(img[0].numpy(), cmap="gray")
    axes[shown][0].set_title("Input")
    axes[shown][1].imshow(mask.numpy(), cmap="gray")
    axes[shown][1].set_title("Ground truth")
    axes[shown][2].imshow(img[0].numpy(), cmap="gray")
    axes[shown][2].imshow(cam, cmap="jet", alpha=0.5)
    axes[shown][2].set_title("Grad-CAM overlay")
    for ax in axes[shown]:
        ax.axis("off")

    shown += 1
    if shown >= 3:
        break

plt.tight_layout()
plt.savefig("gradcam_prostate_sanity_check.png", dpi=120)
print("Saved gradcam_prostate_sanity_check.png -- open it to inspect visually")
