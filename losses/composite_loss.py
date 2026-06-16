# losses/composite_loss.py
import torch
import torch.nn as nn

from .base_losses import SoftDiceLoss
from .boundary_loss import BoundaryLoss


class CompositeCrackLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.87,
        dice_weight: float = 0.13,
        use_boundary: bool = False,
        lambda_boundary: float = 0.3,
        eps: float = 1e-6,
        loss_warmup_epochs: int = 0,
    ):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(eps=1.0)
        self.boundary = BoundaryLoss(eps=eps)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.use_boundary = use_boundary
        self.lambda_boundary = lambda_boundary
        self.loss_warmup_epochs = loss_warmup_epochs

    def forward(self, logits, target, epoch=None):
        target = target.to(dtype=logits.dtype)
        prob = torch.sigmoid(logits)
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(prob, target)
        loss_boundary = logits.new_zeros(())

        boundary_enabled = self.use_boundary and not (
            epoch is not None and epoch < self.loss_warmup_epochs
        )
        if boundary_enabled:
            loss_boundary = self.boundary(prob, target)

        loss_total = (
            self.bce_weight * loss_bce
            + self.dice_weight * loss_dice
            + self.lambda_boundary * loss_boundary
        )
        return {
            "loss_total": loss_total,
            "loss_bce": loss_bce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_boundary": loss_boundary.detach(),
        }


class CompositeLoss(nn.Module):
    def __init__(
        self,
        base_loss,
        boundary_loss=None,
        cldice_loss=None,
        tv_loss=None,
        lambda_boundary=0.3,
        lambda_cldice=0.2,
        lambda_tv=0.02,
        warmup_epochs=10,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = boundary_loss
        self.cldice_loss = cldice_loss
        self.tv_loss = tv_loss
        self.lambda_boundary = lambda_boundary
        self.lambda_cldice = lambda_cldice
        self.lambda_tv = lambda_tv
        self.warmup_epochs = warmup_epochs

    def forward(self, logits, target, image=None, dist_map=None, epoch=0):
        prob = torch.sigmoid(logits)
        out = self.base_loss(logits, target)
        total = out["loss_base"]

        if epoch >= self.warmup_epochs:
            if self.boundary_loss is not None and dist_map is not None:
                lb = self.boundary_loss(prob, dist_map)
                out["loss_boundary"] = lb
                total = total + self.lambda_boundary * lb

            if self.cldice_loss is not None:
                lc = self.cldice_loss(prob, target)
                out["loss_cldice"] = lc
                total = total + self.lambda_cldice * lc

            if self.tv_loss is not None and image is not None:
                lt = self.tv_loss(prob, image)
                out["loss_tv"] = lt
                total = total + self.lambda_tv * lt

        out["loss_total"] = total
        return out
