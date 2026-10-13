"""
Prints the model's layer structure so we can pick the right target layer
for Grad-CAM (typically the last conv block before the bottleneck, or the
last decoder block if you want CAMs that reflect final segmentation detail).

Run:
    python inspect_model.py
"""
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.residual_unet import build_residual_unet

model = build_residual_unet()
print(model)

print("\n--- Named modules (for picking a Grad-CAM target layer) ---")
for name, module in model.named_modules():
    print(name, "->", module.__class__.__name__)
