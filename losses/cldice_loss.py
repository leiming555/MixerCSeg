# losses/cldice_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def soft_erode(x):
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)

def soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

def soft_open(x):
    return soft_dilate(soft_erode(x))

def soft_skel(x, iter_num=10):
    img = x
    skel = torch.relu(img - soft_open(img))
    for _ in range(iter_num):
        img = soft_erode(img)
        delta = torch.relu(img - soft_open(img))
        skel = skel + torch.relu(delta - skel * delta)
    return skel

class SoftCLDiceLoss(nn.Module):
    def __init__(self, iter_num=10, eps=1e-6):
        super().__init__()
        self.iter_num = iter_num
        self.eps = eps

    def forward(self, prob, target):
        skel_p = soft_skel(prob, self.iter_num)
        skel_t = soft_skel(target, self.iter_num)
        tprec = (skel_p * target).sum() / (skel_p.sum() + self.eps)
        tsens = (skel_t * prob).sum() / (skel_t.sum() + self.eps)
        cldice = (2 * tprec * tsens) / (tprec + tsens + self.eps)
        return 1.0 - cldice
