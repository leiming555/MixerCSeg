# losses/smoothness_loss.py
import torch
import torch.nn as nn

class EdgeAwareTVLoss(nn.Module):
    def __init__(self, beta: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, prob: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # image 归一化图像，[B,3,H,W]
        dx_p = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs()
        dy_p = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs()
        dx_i = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean(1, keepdim=True)
        dy_i = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs().mean(1, keepdim=True)
        wx = torch.exp(-self.beta * dx_i)
        wy = torch.exp(-self.beta * dy_i)
        return (wx * dx_p).mean() + (wy * dy_p).mean()
