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

class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        eps: float = 1e-6,
        gamma: float = 1.0,
        multiscale: bool = False,
        tolerant_kernel: int = 1,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.gamma = gamma
        self.multiscale = multiscale
        self.tolerant_kernel = tolerant_kernel

    def _single_scale_loss(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(2, prob.ndim))
        target = target.to(dtype=prob.dtype)
        fp_target = target
        if self.tolerant_kernel > 1:
            with torch.no_grad():
                fp_target = dilate_binary_target(target, self.tolerant_kernel)
        tp = (prob * target).sum(dims)
        fp = (prob * (1.0 - fp_target)).sum(dims)
        fn = ((1.0 - prob) * target).sum(dims)
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss = 1.0 - score
        if self.gamma != 1.0:
            loss = loss.clamp_min(0.0).pow(self.gamma)
        return loss.mean()

    def _downsample(self, prob: torch.Tensor, target: torch.Tensor, scale: int):
        if scale <= 1:
            return prob, target
        prob_s = F.avg_pool2d(
            prob,
            kernel_size=scale,
            stride=scale,
            ceil_mode=True,
            count_include_pad=False,
        )
        target_s = F.max_pool2d(target, kernel_size=scale, stride=scale, ceil_mode=True)
        return prob_s, target_s

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(dtype=prob.dtype)
        if not self.multiscale or prob.ndim != 4:
            return self._single_scale_loss(prob, target)

        losses = []
        for scale in (1, 2, 4):
            prob_s, target_s = self._downsample(prob, target, scale)
            losses.append(self._single_scale_loss(prob_s, target_s))
        return torch.stack(losses).mean()

def dilate_binary_target(target: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return target
    if kernel_size % 2 == 0:
        raise ValueError("dilate_kernel must be an odd positive integer")
    if target.ndim == 4:
        return F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    if target.ndim == 3:
        dilated = F.max_pool2d(target.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return dilated.squeeze(1)
    raise ValueError("target for dilated BCE must be a 3D or 4D tensor")

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
