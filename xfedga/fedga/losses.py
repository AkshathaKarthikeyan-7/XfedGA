"""
Equal-weighted DSC + Cross-Entropy loss, matching the base paper's
implementation detail (Sec VI-E): "minimize the weighted combination of DSC
and Cross-Entropy losses, with equal weighting assigned to both components."

Uses MONAI's DiceCELoss directly rather than hand-rolling it -- it already
implements this exact combination correctly (softmax + one-hot handling for
multi-class output), which matches your model's 2-channel (lesion/background)
softmax output.
"""
from monai.losses import DiceCELoss


def build_loss():
    return DiceCELoss(
        to_onehot_y=True,   # masks come in as (B,H,W) int labels; auto one-hot to (B,2,H,W)
        softmax=True,       # model outputs raw logits; apply softmax internally
        lambda_dice=0.5,    # equal weighting between the two terms
        lambda_ce=0.5,
    )
