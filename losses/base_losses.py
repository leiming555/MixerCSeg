# losses/base_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, prob.ndim))
        inter = (prob * target).sum(dims)
        den = prob.sum(dims) + target.sum(dims)
        dice = (2.0 * inter + self.eps) / (den + self.eps)
        return 1.0 - dice.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5, pos_weight=None, eps: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = SoftDiceLoss(eps=eps)
        self.bce_w = bce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(prob, target)
        return {
            "loss_bce": loss_bce,
            "loss_dice": loss_dice,
            "loss_base": self.bce_w * loss_bce + self.dice_w * loss_dice,
        }
