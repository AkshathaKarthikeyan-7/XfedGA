"""
2D Residual U-Net for breast lesion segmentation.
Matches the base paper's spec: channels [16,32,64,128,256], strides [2,2,2,2],
two residual units per conv block, skip connections, softmax output.

Uses MONAI's UNet with residual units for a faithful, well-tested reproduction
instead of hand-rolling — this also makes swapping in the Attention U-Net
(monai.networks.nets.AttentionUnet) trivial for the prostate task later.
"""
from monai.networks.nets import UNet, AttentionUnet


def build_residual_unet(in_channels=1, out_channels=2):
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_attention_unet(in_channels=1, out_channels=2):
    """For the prostate task (phase 4) — same channel config for comparability."""
    return AttentionUnet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    )
