import sys
from pathlib import Path

from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
import torch
from mmseg.utils import get_root_logger
from mmcv.runner import _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from models.utils import LayerNorm1D, LayerNorm2D, FFN, Stem, PatchMerging
from models.layers import HoGEdgeGateConv
from models.ccem import CrackContinuityEnhancementModule
from models.edrm import EdgeDetailRecoveryModule
from models.experimental_modules import ExperimentalEnhancementModule
from models.experimental_second_modules import SecondExperimentalEnhancementModule
from models.experimental_third_modules import ThirdExperimentalEnhancementModule
from models.experimental_paper_stack import PaperStackEnhancementModule
from models.deg_ridge_scale_gap_interaction import DEGRidgeScaleGapInteractionModule
from models.rsgdi_v2_modules import RSGDIV2EnhancementModule
from models.experimental_triple_stack_v3 import TripleStackV3Block
from models.experimental_triple_stack_v4 import TripleStackV4Block
from models.experimental_triple_stack_v5 import TripleStackV5Block
from models.experimental_triple_stack_v6 import TripleStackV6Block
from models.experimental_triple_stack_v7 import TripleStackV7Block
from models.experimental_triple_stack_v8 import TripleStackV8Block
from models.experimental_triple_stack_v9 import TripleStackV9Block
from models.experimental_triple_stack_v10 import TripleStackV10Block
from models.experimental_triple_stack_v11 import TripleStackV11Block

from VMamba.models.vmamba import TransMixer



class VSS(nn.Module):
    def __init__(
        self,
        in_dim,
        depth,
        mlp_ratio=4.,
        state_dim=64,
        nbins=36,
        use_ccem=False,
        ccem_mode="full",
        ccem_gate_mode="original",
        ccem_branch_weight=False,
        use_edrm=False,
        use_exp_module=False,
        exp_module_mode="win_attn_strip_gate",
        use_exp_second_module=False,
        exp_second_module_mode="soft_morph_gradient_gate",
        use_exp_third_module=False,
        exp_third_module_mode="hessian_eigen_bridge_gate",
        use_paper_stack=False,
        paper_stack_mode="saf_rgp",
        use_drsgi=False,
        drsgi_mode="pre_gate",
        use_rsgdi_v2=False,
        rsgdi_v2_mode="topology_bridge_gate",
        use_triple_stack_v3=False,
        triple_stack_v3_mode="dsc_htm_gbc",
        use_triple_stack_v4=False,
        triple_stack_v4_mode="amc_eov_tgc",
        use_triple_stack_v5=False,
        triple_stack_v5_mode="mwr_eoc_upb",
        use_triple_stack_v6=False,
        triple_stack_v6_mode="hfr_ert_apb",
        use_triple_stack_v7=False,
        triple_stack_v7_mode="rma_mev_dgb",
        use_triple_stack_v8=False,
        triple_stack_v8_mode="cwc_lsf_obp",
        use_triple_stack_v9=False,
        triple_stack_v9_mode="eub_oeb_prg",
        use_triple_stack_v10=False,
        triple_stack_v10_mode="ard_dbr_bpg",
        use_triple_stack_v11=False,
        triple_stack_v11_mode="ram_rcv_cgb",
    ):
        super().__init__()
        if use_triple_stack_v11 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_triple_stack_v6 or use_triple_stack_v7 or use_triple_stack_v8
            or use_triple_stack_v9 or use_triple_stack_v10 or use_exp_module
            or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v11 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v10 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_triple_stack_v6 or use_triple_stack_v7 or use_triple_stack_v8
            or use_triple_stack_v9 or use_exp_module or use_exp_second_module
            or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v10 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v9 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_triple_stack_v6 or use_triple_stack_v7 or use_triple_stack_v8
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v9 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v8 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_triple_stack_v6 or use_triple_stack_v7
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v8 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v7 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_triple_stack_v6 or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v7 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v6 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v6 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v5 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v4
            or use_triple_stack_v6 or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v5 cannot be combined with other experimental module entries"
            )
        if use_triple_stack_v4 and (
            use_ccem or use_edrm or use_drsgi or use_paper_stack or use_rsgdi_v2
            or use_triple_stack_v3 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v4 cannot be combined with other experimental module entries"
            )
        if use_rsgdi_v2 and (
            use_drsgi or use_paper_stack or use_triple_stack_v3
            or use_triple_stack_v4 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_rsgdi_v2 cannot be combined with use_drsgi/use_paper_stack/use_triple_stack_v3/"
                "use_exp_module/use_exp_second_module/use_exp_third_module"
            )
        if use_drsgi and (
            use_paper_stack or use_triple_stack_v3
            or use_triple_stack_v4 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_drsgi cannot be combined with use_paper_stack/use_triple_stack_v3/"
                "use_exp_module/use_exp_second_module/use_exp_third_module"
            )
        if use_paper_stack and (
            use_triple_stack_v3 or use_triple_stack_v4 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_paper_stack cannot be combined with use_triple_stack_v3/use_exp_module/"
                "use_exp_second_module/use_exp_third_module"
            )
        if use_triple_stack_v3 and (
            use_triple_stack_v4 or use_triple_stack_v5
            or use_exp_module or use_exp_second_module or use_exp_third_module
        ):
            raise ValueError(
                "use_triple_stack_v3 cannot be combined with use_exp_module/"
                "use_exp_second_module/use_exp_third_module"
            )
        if use_exp_second_module and (not use_exp_module or exp_module_mode != "scale_adaptive_fusion"):
            raise ValueError(
                "use_exp_second_module requires use_exp_module with exp_module_mode='scale_adaptive_fusion'"
            )
        if use_exp_third_module and (
            not use_exp_module
            or exp_module_mode != "scale_adaptive_fusion"
            or not use_exp_second_module
            or exp_second_module_mode != "ridge_hessian_context"
        ):
            raise ValueError(
                "use_exp_third_module requires scale_adaptive_fusion followed by ridge_hessian_context"
            )
        self.depth = depth
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            layers = [
                TransMixer(hidden_dim=in_dim, ssm_d_state=state_dim, mlp_ratio=mlp_ratio, channel_first=True),
            ]
            if use_triple_stack_v11:
                layers.append(TripleStackV11Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v11_mode,
                ))
            elif use_triple_stack_v10:
                layers.append(TripleStackV10Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v10_mode,
                ))
            elif use_triple_stack_v9:
                layers.append(TripleStackV9Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v9_mode,
                ))
            elif use_triple_stack_v8:
                layers.append(TripleStackV8Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v8_mode,
                ))
            elif use_triple_stack_v7:
                layers.append(TripleStackV7Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v7_mode,
                ))
            elif use_triple_stack_v6:
                layers.append(TripleStackV6Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v6_mode,
                ))
            elif use_triple_stack_v5:
                layers.append(TripleStackV5Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v5_mode,
                ))
            elif use_triple_stack_v4:
                layers.append(TripleStackV4Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v4_mode,
                ))
            elif use_triple_stack_v3:
                layers.append(TripleStackV3Block(
                    channels=in_dim,
                    nbins=nbins,
                    mode=triple_stack_v3_mode,
                ))
            elif use_drsgi:
                layers.append(DEGRidgeScaleGapInteractionModule(
                    channels=in_dim,
                    nbins=nbins,
                    mode=drsgi_mode,
                ))
            else:
                layers.append(HoGEdgeGateConv(
                            in_dim=in_dim,
                            nbins=nbins
                ))
            if use_rsgdi_v2:
                layers.append(RSGDIV2EnhancementModule(
                    channels=in_dim,
                    mode=rsgdi_v2_mode,
                ))
            if use_paper_stack:
                layers.append(PaperStackEnhancementModule(
                    channels=in_dim,
                    mode=paper_stack_mode,
                ))
            if use_edrm:
                layers.append(EdgeDetailRecoveryModule(channels=in_dim))
            if use_exp_module:
                layers.append(ExperimentalEnhancementModule(
                    channels=in_dim,
                    mode=exp_module_mode,
                ))
            if use_exp_second_module:
                layers.append(SecondExperimentalEnhancementModule(
                    channels=in_dim,
                    mode=exp_second_module_mode,
                ))
            if use_exp_third_module:
                layers.append(ThirdExperimentalEnhancementModule(
                    channels=in_dim,
                    mode=exp_third_module_mode,
                ))
            if use_ccem:
                layers.append(CrackContinuityEnhancementModule(
                    channels=in_dim,
                    mode=ccem_mode,
                    gate_mode=ccem_gate_mode,
                    branch_weight=ccem_branch_weight,
                ))
            block = nn.Sequential(*layers)
            self.blocks.append(block)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class VSSEncoder(nn.Module):
    def __init__(self, in_dim=3,  
                 embed_dim=[128,256,512], 
                 depths=[2, 2, 2], 
                 mlp_ratio=4.,
                 state_dim=[49,25,9], distillation=False,
                 is_patch_embed=True,
                 nbins=36,
                 use_ccem=False,
                 ccem_mode="full",
                 ccem_gate_mode="original",
                 ccem_branch_weight=False,
                 use_edrm=False,
                 edrm_stages="f1",
                 use_exp_module=False,
                 exp_module_mode="win_attn_strip_gate",
                 use_exp_second_module=False,
                 exp_second_module_mode="soft_morph_gradient_gate",
                 use_exp_third_module=False,
                 exp_third_module_mode="hessian_eigen_bridge_gate",
                 use_paper_stack=False,
                 paper_stack_mode="saf_rgp",
                 use_drsgi=False,
                 drsgi_mode="pre_gate",
                 use_rsgdi_v2=False,
                 rsgdi_v2_mode="topology_bridge_gate",
                 use_triple_stack_v3=False,
                 triple_stack_v3_mode="dsc_htm_gbc",
                 use_triple_stack_v4=False,
                 triple_stack_v4_mode="amc_eov_tgc",
                 use_triple_stack_v5=False,
                 triple_stack_v5_mode="mwr_eoc_upb",
                 use_triple_stack_v6=False,
                 triple_stack_v6_mode="hfr_ert_apb",
                 use_triple_stack_v7=False,
                 triple_stack_v7_mode="rma_mev_dgb",
                 use_triple_stack_v8=False,
                 triple_stack_v8_mode="cwc_lsf_obp",
                 use_triple_stack_v9=False,
                 triple_stack_v9_mode="eub_oeb_prg",
                 use_triple_stack_v10=False,
                 triple_stack_v10_mode="ard_dbr_bpg",
                 use_triple_stack_v11=False,
                 triple_stack_v11_mode="ram_rcv_cgb",
                 ):
        super().__init__()
        if edrm_stages not in {"f1", "f1_f2"}:
            raise ValueError(f"Unsupported edrm_stages: {edrm_stages}")
        self.num_layers = len(depths)
        self.distillation =distillation
        if is_patch_embed:
            self.patch_embed = Stem(in_dim=in_dim, dim=embed_dim[0])
        self.is_patch_embed = is_patch_embed    

        # build stages
        self.vss_layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            edrm_stage_limit = 2 if edrm_stages == "f1_f2" else 1
            stage_use_edrm = use_edrm and i_layer < edrm_stage_limit

            vss = VSS(in_dim=int(embed_dim[i_layer]),
                      depth=depths[i_layer],
                      mlp_ratio=mlp_ratio,
                      state_dim = state_dim[i_layer],
                      nbins=nbins,
                      use_ccem=use_ccem,
                      ccem_mode=ccem_mode,
                      ccem_gate_mode=ccem_gate_mode,
                      ccem_branch_weight=ccem_branch_weight,
                      use_edrm=stage_use_edrm,
                      use_exp_module=use_exp_module,
                      exp_module_mode=exp_module_mode,
                      use_exp_second_module=use_exp_second_module,
                      exp_second_module_mode=exp_second_module_mode,
                      use_exp_third_module=use_exp_third_module,
                      exp_third_module_mode=exp_third_module_mode,
                      use_paper_stack=use_paper_stack,
                      paper_stack_mode=paper_stack_mode,
                      use_drsgi=use_drsgi,
                      drsgi_mode=drsgi_mode,
                      use_rsgdi_v2=use_rsgdi_v2,
                      rsgdi_v2_mode=rsgdi_v2_mode,
                      use_triple_stack_v3=use_triple_stack_v3,
                      triple_stack_v3_mode=triple_stack_v3_mode,
                      use_triple_stack_v4=use_triple_stack_v4,
                      triple_stack_v4_mode=triple_stack_v4_mode,
                      use_triple_stack_v5=use_triple_stack_v5,
                      triple_stack_v5_mode=triple_stack_v5_mode,
                      use_triple_stack_v6=use_triple_stack_v6,
                      triple_stack_v6_mode=triple_stack_v6_mode,
                      use_triple_stack_v7=use_triple_stack_v7,
                      triple_stack_v7_mode=triple_stack_v7_mode,
                      use_triple_stack_v8=use_triple_stack_v8,
                      triple_stack_v8_mode=triple_stack_v8_mode,
                      use_triple_stack_v9=use_triple_stack_v9,
                      triple_stack_v9_mode=triple_stack_v9_mode,
                      use_triple_stack_v10=use_triple_stack_v10,
                      triple_stack_v10_mode=triple_stack_v10_mode,
                      use_triple_stack_v11=use_triple_stack_v11,
                      triple_stack_v11_mode=triple_stack_v11_mode)
            self.vss_layers.append(vss)

            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(in_dim=int(embed_dim[i_layer]), 
                                          out_dim=int(embed_dim[i_layer+1])) 
                self.downsamples.append(downsample)

        self.apply(self._init_weights)
        self = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
        self.train()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm2D):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            logger.info(f"Miss {missing_keys}")
            logger.info(f"Unexpected {unexpected_keys}")

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VSSEncoder, self).train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        if self.is_patch_embed:
            x = self.patch_embed(x)
            
        outs = []
        for i in range(self.num_layers):
            vss = self.vss_layers[i]
            x = vss(x)
        
            outs.append(x)     
            if i < self.num_layers - 1:
                down = self.downsamples[i]
                x = down(x)

        return outs
    
