import torch
from torch import nn
from losses.base_losses import TverskyLoss, dilate_binary_target
from models.decoder import SRFModule
from models.encoder import VSSEncoder



class MixerCSeg(nn.Module):
    def __init__(self, backbone, embed_dims, args=None):
        super().__init__()
        self.args = args
        self.backbone = backbone    
        self.decoder = SRFModule(embed_dims, mid_dim=8, size=(args.load_width, args.load_height))

    def forward(self, samples):
        outs = self.backbone(samples)
        out = self.decoder(outs)

        return out

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self, args):
        super(bce_dice, self).__init__()
        pos_weight = getattr(args, 'pos_weight', 1.0)
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight > 1.0 else None
        self.bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice_fn = DiceLoss()
        self.tversky_fn = TverskyLoss(
            alpha=getattr(args, 'tversky_alpha', 0.3),
            beta=getattr(args, 'tversky_beta', 0.7),
            eps=getattr(args, 'eps', 1e-6),
        )
        self.args = args
        self.use_tversky = getattr(args, 'use_tversky', False)
        self.lambda_tversky = getattr(args, 'lambda_tversky', 0.2)
        self.use_dilated_bce = getattr(args, 'use_dilated_bce', False)
        self.dilate_kernel = getattr(args, 'dilate_kernel', 3)

    def forward(self, y_pred, y_true):
        y_true = y_true.to(dtype=y_pred.dtype)
        bce_target = y_true
        if self.use_dilated_bce:
            bce_target = dilate_binary_target(y_true, self.dilate_kernel)
        prob = y_pred.sigmoid()
        bce = self.bce_fn(y_pred, bce_target)
        dice = self.dice_fn(prob, y_true)
        loss = self.args.BCELoss_ratio * bce + self.args.DiceLoss_ratio * dice
        if self.use_tversky:
            loss = loss + self.lambda_tversky * self.tversky_fn(prob, y_true)
        return loss



def build_MixerCSeg(args):
    device = torch.device(args.device)
    args.device = torch.device(args.device)

    embed_dim=[16,32,64,128]

    depths = [1,1,1,1]
    state_dim=[8,8,16,16]

    backbone = VSSEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        depths=depths,
        mlp_ratio=2.,
        state_dim=state_dim,
        nbins=getattr(args, 'nbins', 36),
        use_ccem=getattr(args, 'use_ccem', False),
        ccem_mode=getattr(args, 'ccem_mode', 'full'),
        ccem_gate_mode=getattr(args, 'ccem_gate_mode', 'original'),
        ccem_branch_weight=getattr(args, 'ccem_branch_weight', False),
        use_edrm=getattr(args, 'use_edrm', False),
        edrm_stages=getattr(args, 'edrm_stages', 'f1'),
        )
    model = MixerCSeg(backbone, embed_dim, args).to(device)

    criterion = bce_dice(args)
    criterion.to(device)
    
    return model, criterion
