# losses/boundary_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_losses import SoftDiceLoss


class BoundaryLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dice = SoftDiceLoss(eps=eps)
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
            ).view(1, 1, 3, 3),
        )

    def _sobel(self, value: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        kernel_x = self.sobel_x.to(dtype=value.dtype).repeat(channels, 1, 1, 1)
        kernel_y = self.sobel_y.to(dtype=value.dtype).repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(value, kernel_x, padding=1, groups=channels)
        grad_y = F.conv2d(value, kernel_y, padding=1, groups=channels)
        magnitude = torch.sqrt(grad_x.square() + grad_y.square() + self.eps)
        magnitude = magnitude - self.eps ** 0.5
        return torch.tanh(magnitude)

    def forward(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_boundary = self._sobel(prob)
        with torch.no_grad():
            gt_boundary = self._sobel(target.to(dtype=prob.dtype))
        return self.dice(pred_boundary, gt_boundary)
