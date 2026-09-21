import os
import argparse
import datetime
import random
import time
import json
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from engine import train_one_epoch
from models import build_MixerCSeg as build_model
from datasets import create_dataset
import cv2
from eval.evaluate import eval
from util.logger import get_logger
from tqdm import tqdm
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES
from models.experimental_paper_stack import PAPER_STACK_MODES
from models.deg_ridge_scale_gap_interaction import DRSGI_MODES
from models.rsgdi_v2_modules import RSGDI_V2_MODES
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES
from models.experimental_triple_stack_v7 import TRIPLE_STACK_V7_MODES
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES
from models.experimental_triple_stack_v9 import TRIPLE_STACK_V9_MODES
from models.experimental_triple_stack_v10 import TRIPLE_STACK_V10_MODES
from models.experimental_triple_stack_v11 import TRIPLE_STACK_V11_MODES

# from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('MixerCSeg FOR CRACK', add_help=False)

    parser.add_argument('--validation_selection', action='store_true',
                        help='Select checkpoints on val with standard sigmoid metrics; never evaluate test during training')

    parser.add_argument('--BCELoss_ratio', default=1.0, type=float,
                        help='Weight for Binary Cross Entropy Loss')
    parser.add_argument('--DiceLoss_ratio', default=5.0, type=float,
                        help='Weight for Dice Loss')
    parser.add_argument('--use_tversky', action='store_true',
                        help='Enable BCE + Dice + Tversky training loss')
    parser.add_argument('--lambda_tversky', default=0.2, type=float,
                        help='Weight for Tversky Loss')
    parser.add_argument('--tversky_alpha', default=0.3, type=float,
                        help='Alpha for Tversky Loss false-positive penalty')
    parser.add_argument('--tversky_beta', default=0.7, type=float,
                        help='Beta for Tversky Loss false-negative penalty')
    parser.add_argument('--tversky_gamma', default=1.0, type=float,
                        help='Focal exponent for Tversky Loss; 1.0 keeps the original Tversky form')
    parser.add_argument('--tversky_multiscale', action='store_true',
                        help='Apply Tversky Loss at 1x, 1/2x, and 1/4x scales')
    parser.add_argument('--tversky_tolerant_kernel', default=1, type=int,
                        help='Odd dilation kernel for Tversky false-positive tolerance; 1 disables tolerance')
    parser.add_argument('--pos_weight', default=1.0, type=float,
                        help='Positive-class weight for BCEWithLogitsLoss; enabled when > 1')
    parser.add_argument('--use_dilated_bce', action='store_true',
                        help='Use dilated masks for BCE target only')
    parser.add_argument('--dilate_kernel', default=3, type=int,
                        help='Odd kernel size for dilated BCE mask')
    parser.add_argument('--use_boundary_loss', action='store_true',
                        help='Enable BCE + Dice + Boundary training loss')
    parser.add_argument('--lambda_boundary', default=0.3, type=float,
                        help='Weight for Boundary Loss')
    parser.add_argument('--loss_warmup_epochs', default=0, type=int,
                        help='Number of initial epochs without Boundary Loss')
    parser.add_argument('--eps', default=1e-6, type=float,
                        help='Numerical stability epsilon for auxiliary losses')
    parser.add_argument('--Norm_Type', default='GN', type=str,
                        help='Normalization layer type [GN|BN], GN=GroupNorm')
    parser.add_argument('--nbins', default=36, type=int,
                        help='Number of direction intervals used by DEGConv')
    parser.add_argument('--use_ccem', action='store_true',
                        help='Enable CCEM after DEGConv in every encoder stage')
    parser.add_argument('--ccem_mode', default='full', type=str,
                        choices=['full', 'enhanced', 'transformer', 'no_local', 'no_strip', 'no_dilation', 'no_gate'],
                        help='CCEM ablation mode')
    parser.add_argument('--ccem_gate_mode', default='original', type=str,
                        choices=['original', 'leaky'],
                        help='CCEM gate mode')
    parser.add_argument('--ccem_branch_weight', action='store_true',
                        help='Enable learnable scalar weights for CCEM branches')
    parser.add_argument('--use_edrm', action='store_true',
                        help='Enable EDRM after DEGConv and before CCEM in shallow encoder stages')
    parser.add_argument('--edrm_stages', default='f1', type=str,
                        choices=['f1', 'f1_f2'],
                        help='EDRM placement: f1 or f1_f2')
    parser.add_argument('--use_exp_module', action='store_true',
                        help='Enable experimental enhancement module after DEGConv')
    parser.add_argument('--exp_module_mode', default='win_attn_strip_gate', type=str,
                        choices=EXPERIMENTAL_MODULE_MODES,
                        help='Experimental enhancement module mode')
    parser.add_argument('--use_exp_second_module', action='store_true',
                        help='Enable second experimental module after scale_adaptive_fusion')
    parser.add_argument('--exp_second_module_mode', default='soft_morph_gradient_gate', type=str,
                        choices=SECOND_EXPERIMENTAL_MODULE_MODES,
                        help='Second experimental enhancement module mode')
    parser.add_argument('--use_exp_third_module', action='store_true',
                        help='Enable third experimental module after scale_adaptive_fusion and ridge_hessian_context')
    parser.add_argument('--exp_third_module_mode', default='hessian_eigen_bridge_gate', type=str,
                        choices=THIRD_EXPERIMENTAL_MODULE_MODES,
                        help='Third experimental enhancement module mode')
    parser.add_argument('--use_paper_stack', action='store_true',
                        help='Enable paper stack enhancement module after DEGConv')
    parser.add_argument('--paper_stack_mode', default='saf_rgp', type=str,
                        choices=PAPER_STACK_MODES,
                        help='Paper stack enhancement module mode')
    parser.add_argument('--use_drsgi', action='store_true',
                        help='Enable direction-guided ridge-scale-gap interaction block in place of DEGConv')
    parser.add_argument('--drsgi_mode', default='pre_gate', type=str,
                        choices=DRSGI_MODES,
                        help='DRSGI interaction mode')
    parser.add_argument('--use_rsgdi_v2', action='store_true',
                        help='Enable RSGDI-v2 after DEGConv, using saf_rgp_rgd as the base interaction stack')
    parser.add_argument('--rsgdi_v2_mode', default='topology_bridge_gate', type=str,
                        choices=RSGDI_V2_MODES,
                        help='RSGDI-v2 enhancement mode')
    parser.add_argument('--use_triple_stack_v3', action='store_true',
                        help='Enable TripleStack-v3 with three fixed innovation points around DEGConv')
    parser.add_argument('--triple_stack_v3_mode', default='dsc_htm_gbc', type=str,
                        choices=TRIPLE_STACK_V3_MODES,
                        help='TripleStack-v3 mode')
    parser.add_argument('--use_triple_stack_v4', action='store_true',
                        help='Enable TripleStack-v4 with adaptive scale, ridge orientation, and gap calibration')
    parser.add_argument('--triple_stack_v4_mode', default='amc_eov_tgc', type=str,
                        choices=TRIPLE_STACK_V4_MODES,
                        help='TripleStack-v4 mode')
    parser.add_argument('--use_triple_stack_v5', action='store_true',
                        help='Enable TripleStack-v5 with width-scale, ridge-topology, and boundary calibration')
    parser.add_argument('--triple_stack_v5_mode', default='mwr_eoc_upb', type=str,
                        choices=TRIPLE_STACK_V5_MODES,
                        help='TripleStack-v5 mode')
    parser.add_argument('--use_triple_stack_v6', action='store_true',
                        help='Enable TripleStack-v6 with composite scale, topology, and boundary modules')
    parser.add_argument('--triple_stack_v6_mode', default='hfr_ert_apb', type=str,
                        choices=TRIPLE_STACK_V6_MODES,
                        help='TripleStack-v6 mode')
    parser.add_argument('--use_triple_stack_v7', action='store_true',
                        help='Enable TripleStack-v7 with base-preserving MSA, OCV, and GBC refinements')
    parser.add_argument('--triple_stack_v7_mode', default='rma_mev_dgb', type=str,
                        choices=TRIPLE_STACK_V7_MODES,
                        help='TripleStack-v7 mode')
    parser.add_argument('--use_triple_stack_v8', action='store_true',
                        help='Enable direction-field-coupled TripleStack-v8 around DEGConv')
    parser.add_argument('--triple_stack_v8_mode', default='cwc_lsf_obp', type=str,
                        choices=TRIPLE_STACK_V8_MODES,
                        help='TripleStack-v8 mode')
    parser.add_argument('--use_triple_stack_v9', action='store_true',
                        help='Enable V7-preserving recall-calibrated TripleStack-v9 around DEGConv')
    parser.add_argument('--triple_stack_v9_mode', default='eub_oeb_prg', type=str,
                        choices=TRIPLE_STACK_V9_MODES,
                        help='TripleStack-v9 mode')
    parser.add_argument('--use_triple_stack_v10', action='store_true',
                        help='Enable V7-anchored sidecar-calibrated TripleStack-v10 around DEGConv')
    parser.add_argument('--triple_stack_v10_mode', default='ard_dbr_bpg', type=str,
                        choices=TRIPLE_STACK_V10_MODES,
                        help='TripleStack-v10 mode')
    parser.add_argument('--use_triple_stack_v11', action='store_true',
                        help='Enable reliability-coupled TripleStack-v11 around DEGConv')
    parser.add_argument('--triple_stack_v11_mode', default='ram_rcv_cgb', type=str,
                        choices=TRIPLE_STACK_V11_MODES,
                        help='TripleStack-v11 mode')
    parser.add_argument('--dataset_path', default="/home/linux/code/sod/dataset/CrackMap",
                        help='Root directory path for dataset')
    parser.add_argument('--batch_size_train', type=int, default=1,
                        help='Number of samples per training batch (affects memory usage)')
    parser.add_argument('--batch_size_test', type=int, default=1,
                        help='Number of samples per batch')
    parser.add_argument('--lr_scheduler', type=str, default='PolyLR',
                        help='Learning rate scheduler type [PolyLR|StepLR|CosLR]')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Initial learning rate (base value for schedulers)')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate for PolyLR')
    parser.add_argument('--weight_decay', default=0.01, type=float,
                        help='Weight decay coefficient for regularization')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Total number of training epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Manual epoch number to start training (useful for resuming)')
    parser.add_argument('--lr_drop', default=30, type=int,
                        help='Epoch interval for dropping learning rate in StepLR scheduler')
    parser.add_argument('--sgd', action='store_true',
                        help='Use SGD optimizer instead of default AdamW')
    parser.add_argument('--output_dir', default='./checkpoints/weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--device', default='cuda',
                        help='Computation device [cuda|cpu] for training/inference')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--dataset_mode', type=str, default='crack',
                        help='Dataset mode selector')
    parser.add_argument('--serial_batches', action='store_true',
                        help='Disable random shuffling and use sequential batch sampling if enabled')
    parser.add_argument('--num_threads', default=1, type=int,
                        help='Number of subprocesses for data loading')
    parser.add_argument('--phase', type=str, default='train',
                        help='Runtime phase selector')
    parser.add_argument('--load_width', type=int, default=512,
                        help='Input image width for preprocessing (will be resized)')
    parser.add_argument('--load_height', type=int, default=512,
                        help='Input image height for preprocessing (will be resized)')
    return parser

def main(args):
    if args.use_triple_stack_v11 and (
        args.use_ccem or args.use_tversky or args.use_boundary_loss or args.use_edrm
        or args.use_paper_stack or args.use_drsgi or args.use_rsgdi_v2
        or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_triple_stack_v6
        or args.use_triple_stack_v7 or args.use_triple_stack_v8
        or args.use_triple_stack_v9 or args.use_triple_stack_v10
        or args.use_exp_module or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v11 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v10 and (
        args.use_ccem or args.use_tversky or args.use_boundary_loss or args.use_edrm
        or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_triple_stack_v6 or args.use_triple_stack_v7
        or args.use_triple_stack_v8 or args.use_triple_stack_v9 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v10 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v9 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_triple_stack_v6 or args.use_triple_stack_v7
        or args.use_triple_stack_v8 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v9 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v8 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_triple_stack_v6 or args.use_triple_stack_v7
        or args.use_exp_module or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v8 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v7 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_triple_stack_v6 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v7 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v6 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v5 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v6 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v5 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v4
        or args.use_triple_stack_v6 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v5 cannot be combined with other experimental module entries"
        )
    if args.use_triple_stack_v4 and (
        args.use_ccem or args.use_edrm or args.use_paper_stack or args.use_drsgi
        or args.use_rsgdi_v2 or args.use_triple_stack_v3 or args.use_triple_stack_v5 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v4 cannot be combined with other experimental module entries"
        )
    if args.use_rsgdi_v2 and (
        args.use_drsgi or args.use_paper_stack or args.use_triple_stack_v3
        or args.use_triple_stack_v4 or args.use_triple_stack_v5
        or args.use_exp_module or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_rsgdi_v2 cannot be combined with --use_drsgi/--use_paper_stack/--use_triple_stack_v3/"
            "--use_exp_module/--use_exp_second_module/--use_exp_third_module"
        )
    if args.use_drsgi and (
        args.use_paper_stack or args.use_triple_stack_v3
        or args.use_triple_stack_v4 or args.use_triple_stack_v5
        or args.use_exp_module or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_drsgi cannot be combined with --use_paper_stack/--use_triple_stack_v3/--use_exp_module/"
            "--use_exp_second_module/--use_exp_third_module"
        )
    if args.use_paper_stack and (
        args.use_triple_stack_v3 or args.use_triple_stack_v4 or args.use_triple_stack_v5
        or args.use_exp_module or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_paper_stack cannot be combined with --use_triple_stack_v3/--use_exp_module/"
            "--use_exp_second_module/--use_exp_third_module"
        )
    if args.use_triple_stack_v3 and (
        args.use_triple_stack_v4 or args.use_triple_stack_v5 or args.use_exp_module
        or args.use_exp_second_module or args.use_exp_third_module
    ):
        raise ValueError(
            "use_triple_stack_v3 cannot be combined with --use_exp_module/"
            "--use_exp_second_module/--use_exp_third_module"
        )
    if args.use_exp_second_module and (not args.use_exp_module or args.exp_module_mode != 'scale_adaptive_fusion'):
        raise ValueError(
            "use_exp_second_module requires --use_exp_module --exp_module_mode scale_adaptive_fusion"
        )
    if args.use_exp_third_module and (
        not args.use_exp_module
        or args.exp_module_mode != 'scale_adaptive_fusion'
        or not args.use_exp_second_module
        or args.exp_second_module_mode != 'ridge_hessian_context'
    ):
        raise ValueError(
            "use_exp_third_module requires --use_exp_module --exp_module_mode scale_adaptive_fusion "
            "--use_exp_second_module --exp_second_module_mode ridge_hessian_context"
        )

    checkpoints_path = "../newcseg/checkpoints"
    cur_time = time.strftime('%Y_%m_%d_%H:%M:%S', time.localtime(time.time()))
    dataset_name = (args.dataset_path).split('/')[-1]
    ccem_name = args.ccem_mode if args.use_ccem else 'baseline'
    experiment_name = f'{dataset_name}_seed{args.seed}_ccem_{ccem_name}'
    if args.use_ccem and args.ccem_gate_mode != 'original':
        experiment_name = f'{experiment_name}_gate_{args.ccem_gate_mode}'
    if args.use_ccem and args.ccem_branch_weight:
        experiment_name = f'{experiment_name}_branch_weight'
    if args.use_edrm:
        experiment_name = f'{experiment_name}_edrm_{args.edrm_stages}'
    if args.use_paper_stack:
        experiment_name = f'{experiment_name}_paper_{args.paper_stack_mode}'
    if args.use_drsgi:
        experiment_name = f'{experiment_name}_drsgi_{args.drsgi_mode}'
    if args.use_rsgdi_v2:
        experiment_name = f'{experiment_name}_rsgdi_v2_{args.rsgdi_v2_mode}'
    if args.use_triple_stack_v3:
        experiment_name = f'{experiment_name}_triple_stack_v3_{args.triple_stack_v3_mode}'
    if args.use_triple_stack_v4:
        experiment_name = f'{experiment_name}_triple_stack_v4_{args.triple_stack_v4_mode}'
    if args.use_triple_stack_v5:
        experiment_name = f'{experiment_name}_triple_stack_v5_{args.triple_stack_v5_mode}'
    if args.use_triple_stack_v6:
        experiment_name = f'{experiment_name}_triple_stack_v6_{args.triple_stack_v6_mode}'
    if args.use_triple_stack_v7:
        experiment_name = f'{experiment_name}_triple_stack_v7_{args.triple_stack_v7_mode}'
    if args.use_triple_stack_v8:
        experiment_name = f'{experiment_name}_triple_stack_v8_{args.triple_stack_v8_mode}'
    if args.use_triple_stack_v9:
        experiment_name = f'{experiment_name}_triple_stack_v9_{args.triple_stack_v9_mode}'
    if args.use_triple_stack_v10:
        experiment_name = f'{experiment_name}_triple_stack_v10_{args.triple_stack_v10_mode}'
    if args.use_triple_stack_v11:
        experiment_name = f'{experiment_name}_triple_stack_v11_{args.triple_stack_v11_mode}'
    if args.use_exp_module:
        experiment_name = f'{experiment_name}_exp_{args.exp_module_mode}'
    if args.use_exp_second_module:
        experiment_name = f'{experiment_name}_second_{args.exp_second_module_mode}'
    if args.use_exp_third_module:
        experiment_name = f'{experiment_name}_third_{args.exp_third_module_mode}'
    process_folder_path = os.path.join(checkpoints_path, cur_time + '_' + experiment_name)
    args.phase = 'train'
    if not os.path.exists(process_folder_path):
        os.makedirs(process_folder_path)
    else:
        print("create process folder error!")

    log_train = get_logger(process_folder_path, experiment_name + '_train')
    log_test = get_logger(process_folder_path, experiment_name + '_test')
    log_eval = get_logger(process_folder_path, experiment_name + '_eval')

    # writer = SummaryWriter('./tensorboard_runs/{}'.format(os.path.join(cur_time + '_' + dataset_name)))

    log_train.info("args -> " + str(args))
    log_train.info("args: dataset -> " + str(args.dataset_path))
    log_train.info("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    log_train.info("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))
    log_train.info("args: use_tversky -> " + str(args.use_tversky))
    log_train.info("args: lambda_tversky -> " + str(args.lambda_tversky))
    log_train.info("args: tversky_alpha -> " + str(args.tversky_alpha))
    log_train.info("args: tversky_beta -> " + str(args.tversky_beta))
    log_train.info("args: tversky_gamma -> " + str(args.tversky_gamma))
    log_train.info("args: tversky_multiscale -> " + str(args.tversky_multiscale))
    log_train.info("args: tversky_tolerant_kernel -> " + str(args.tversky_tolerant_kernel))
    log_train.info("args: pos_weight -> " + str(args.pos_weight))
    log_train.info("args: use_dilated_bce -> " + str(args.use_dilated_bce))
    log_train.info("args: dilate_kernel -> " + str(args.dilate_kernel))
    log_train.info("args: use_ccem -> " + str(args.use_ccem))
    log_train.info("args: ccem_mode -> " + str(ccem_name))
    log_train.info("args: ccem_gate_mode -> " + str(args.ccem_gate_mode))
    log_train.info("args: ccem_branch_weight -> " + str(args.ccem_branch_weight))
    log_train.info("args: use_edrm -> " + str(args.use_edrm))
    log_train.info("args: edrm_stages -> " + str(args.edrm_stages))
    log_train.info("args: use_exp_module -> " + str(args.use_exp_module))
    log_train.info("args: exp_module_mode -> " + str(args.exp_module_mode))
    log_train.info("args: use_exp_second_module -> " + str(args.use_exp_second_module))
    log_train.info("args: exp_second_module_mode -> " + str(args.exp_second_module_mode))
    log_train.info("args: use_exp_third_module -> " + str(args.use_exp_third_module))
    log_train.info("args: exp_third_module_mode -> " + str(args.exp_third_module_mode))
    log_train.info("args: use_paper_stack -> " + str(args.use_paper_stack))
    log_train.info("args: paper_stack_mode -> " + str(args.paper_stack_mode))
    log_train.info("args: use_drsgi -> " + str(args.use_drsgi))
    log_train.info("args: drsgi_mode -> " + str(args.drsgi_mode))
    log_train.info("args: use_rsgdi_v2 -> " + str(args.use_rsgdi_v2))
    log_train.info("args: rsgdi_v2_mode -> " + str(args.rsgdi_v2_mode))
    log_train.info("args: use_triple_stack_v3 -> " + str(args.use_triple_stack_v3))
    log_train.info("args: triple_stack_v3_mode -> " + str(args.triple_stack_v3_mode))
    log_train.info("args: use_triple_stack_v4 -> " + str(args.use_triple_stack_v4))
    log_train.info("args: triple_stack_v4_mode -> " + str(args.triple_stack_v4_mode))
    log_train.info("args: use_triple_stack_v5 -> " + str(args.use_triple_stack_v5))
    log_train.info("args: triple_stack_v5_mode -> " + str(args.triple_stack_v5_mode))
    log_train.info("args: use_triple_stack_v6 -> " + str(args.use_triple_stack_v6))
    log_train.info("args: triple_stack_v6_mode -> " + str(args.triple_stack_v6_mode))
    log_train.info("args: use_triple_stack_v7 -> " + str(args.use_triple_stack_v7))
    log_train.info("args: triple_stack_v7_mode -> " + str(args.triple_stack_v7_mode))
    log_train.info("args: use_triple_stack_v8 -> " + str(args.use_triple_stack_v8))
    log_train.info("args: triple_stack_v8_mode -> " + str(args.triple_stack_v8_mode))
    log_train.info("args: use_triple_stack_v9 -> " + str(args.use_triple_stack_v9))
    log_train.info("args: triple_stack_v9_mode -> " + str(args.triple_stack_v9_mode))
    log_train.info("args: use_triple_stack_v10 -> " + str(args.use_triple_stack_v10))
    log_train.info("args: triple_stack_v10_mode -> " + str(args.triple_stack_v10_mode))
    log_train.info("args: use_triple_stack_v11 -> " + str(args.use_triple_stack_v11))
    log_train.info("args: triple_stack_v11_mode -> " + str(args.triple_stack_v11_mode))
    print("args: BCELoss_ratio -> " + str(args.BCELoss_ratio))
    print("args: DiceLoss_ratio -> " + str(args.DiceLoss_ratio))
    print("args: use_tversky -> " + str(args.use_tversky))
    print("args: lambda_tversky -> " + str(args.lambda_tversky))
    print("args: tversky_alpha -> " + str(args.tversky_alpha))
    print("args: tversky_beta -> " + str(args.tversky_beta))
    print("args: tversky_gamma -> " + str(args.tversky_gamma))
    print("args: tversky_multiscale -> " + str(args.tversky_multiscale))
    print("args: tversky_tolerant_kernel -> " + str(args.tversky_tolerant_kernel))
    print("args: pos_weight -> " + str(args.pos_weight))
    print("args: use_dilated_bce -> " + str(args.use_dilated_bce))
    print("args: dilate_kernel -> " + str(args.dilate_kernel))
    print("args: use_ccem -> " + str(args.use_ccem))
    print("args: ccem_mode -> " + str(ccem_name))
    print("args: ccem_gate_mode -> " + str(args.ccem_gate_mode))
    print("args: ccem_branch_weight -> " + str(args.ccem_branch_weight))
    print("args: use_edrm -> " + str(args.use_edrm))
    print("args: edrm_stages -> " + str(args.edrm_stages))
    print("args: use_exp_module -> " + str(args.use_exp_module))
    print("args: exp_module_mode -> " + str(args.exp_module_mode))
    print("args: use_exp_second_module -> " + str(args.use_exp_second_module))
    print("args: exp_second_module_mode -> " + str(args.exp_second_module_mode))
    print("args: use_exp_third_module -> " + str(args.use_exp_third_module))
    print("args: exp_third_module_mode -> " + str(args.exp_third_module_mode))
    print("args: use_paper_stack -> " + str(args.use_paper_stack))
    print("args: paper_stack_mode -> " + str(args.paper_stack_mode))
    print("args: use_drsgi -> " + str(args.use_drsgi))
    print("args: drsgi_mode -> " + str(args.drsgi_mode))
    print("args: use_rsgdi_v2 -> " + str(args.use_rsgdi_v2))
    print("args: rsgdi_v2_mode -> " + str(args.rsgdi_v2_mode))
    print("args: use_triple_stack_v3 -> " + str(args.use_triple_stack_v3))
    print("args: triple_stack_v3_mode -> " + str(args.triple_stack_v3_mode))
    print("args: use_triple_stack_v4 -> " + str(args.use_triple_stack_v4))
    print("args: triple_stack_v4_mode -> " + str(args.triple_stack_v4_mode))
    print("args: use_triple_stack_v5 -> " + str(args.use_triple_stack_v5))
    print("args: triple_stack_v5_mode -> " + str(args.triple_stack_v5_mode))
    print("args: use_triple_stack_v6 -> " + str(args.use_triple_stack_v6))
    print("args: triple_stack_v6_mode -> " + str(args.triple_stack_v6_mode))
    print("args: use_triple_stack_v7 -> " + str(args.use_triple_stack_v7))
    print("args: triple_stack_v7_mode -> " + str(args.triple_stack_v7_mode))
    print("args: use_triple_stack_v8 -> " + str(args.use_triple_stack_v8))
    print("args: triple_stack_v8_mode -> " + str(args.triple_stack_v8_mode))
    print("args: use_triple_stack_v9 -> " + str(args.use_triple_stack_v9))
    print("args: triple_stack_v9_mode -> " + str(args.triple_stack_v9_mode))
    print("args: use_triple_stack_v10 -> " + str(args.use_triple_stack_v10))
    print("args: triple_stack_v10_mode -> " + str(args.triple_stack_v10_mode))
    print("args: use_triple_stack_v11 -> " + str(args.use_triple_stack_v11))
    print("args: triple_stack_v11_mode -> " + str(args.triple_stack_v11_mode))

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args)
    test_criterion = criterion
    if args.use_boundary_loss:
        from losses import CompositeCrackLoss
        criterion = CompositeCrackLoss(
            bce_weight=args.BCELoss_ratio,
            dice_weight=args.DiceLoss_ratio,
            use_boundary=True,
            lambda_boundary=args.lambda_boundary,
            loss_warmup_epochs=args.loss_warmup_epochs,
            eps=args.eps,
            use_tversky=args.use_tversky,
            lambda_tversky=args.lambda_tversky,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            tversky_gamma=args.tversky_gamma,
            tversky_multiscale=args.tversky_multiscale,
            tversky_tolerant_kernel=args.tversky_tolerant_kernel,
            pos_weight=args.pos_weight,
            use_dilated_bce=args.use_dilated_bce,
            dilate_kernel=args.dilate_kernel,
        ).to(device)
    model.to(device)
    args.batch_size = args.batch_size_train
    train_dataLoader = create_dataset(args)
    validation_loader = None
    if getattr(args, 'validation_selection', False):
        from util.standard_validation import make_validation_loader, evaluate_validation, selection_key
        validation_loader = make_validation_loader(args)
        log_train.info('args: validation_selection -> True; selection_split -> val; threshold -> 0.5')
        print('args: validation_selection -> True; selection_split -> val; threshold -> 0.5')
    dataset_size = len(train_dataLoader)
    print('The number of training images = %d' % dataset_size)
    log_train.info('The number of training images = %d' % dataset_size)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()],
            "lr": args.lr,
        },
    ]
    if args.sgd:
        print('use SGD!')
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        print('use AdamW!')
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    output_dir = os.path.join(args.output_dir, cur_time + '_' + experiment_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    print("Start processing! ")
    log_train.info("Start processing! ")
    start_time = time.time()
    max_mIoU = 0
    best_validation_key = None
    best_validation = None
    max_Metrics = {'epoch': 0, 'mIoU': 0, 'ODS': 0, 'OIS': 0, 'F1': 0, 'Precision': 0, 'Recall': 0}

    for epoch in range(args.start_epoch, args.epochs):
        print("---------------------------------------------------------------------------------------")
        print("training epoch start -> ", epoch)
        train_one_epoch(model, criterion, train_dataLoader, optimizer, epoch, args, log_train)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print("training epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        if validation_loader is not None:
            metrics = evaluate_validation(model, validation_loader, device)
            record = dict(metrics, epoch=epoch, split='val', fixed_threshold=0.5)
            with (output_dir / 'validation_history.jsonl').open('a') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            key = selection_key(metrics, epoch)
            if best_validation_key is None or key > best_validation_key:
                best_validation_key = key
                best_validation = record
                utils.save_on_master({
                    'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(), 'epoch': epoch,
                    'args': args, 'selection': record,
                }, output_dir / 'checkpoint_best_val.pth')
            print('Validation metrics | ' + json.dumps(record, allow_nan=False))
            log_eval.info('Validation metrics | ' + json.dumps(record, allow_nan=False))
            continue

        print("testing epoch start -> ", epoch)
        results_path = cur_time + '_' + experiment_name
        save_root = f'./results/{results_path}/results_' + str(epoch)
        args.phase = 'test'
        args.batch_size = args.batch_size_test
        test_dl = create_dataset(args)
        pbar = tqdm(total=len(test_dl), desc=f"Initial Loss: Pending")

        if not os.path.isdir(save_root):
            os.makedirs(save_root)
        with torch.no_grad():
            model.eval()
            for batch_idx, (data) in enumerate(test_dl):
                x = data["image"]
                target = data["label"]
                if device != 'cpu':
                    x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
                out = model(x)
                loss = test_criterion(out, target.float())
                target = target[0, 0, ...].cpu().numpy()
                out = out[0, 0, ...].cpu().numpy()
                root_name = data["A_paths"][0].split("/")[-1][0:-4]

                target = 255 * (target / np.max(target))
                out = 255 * (out / np.max(out))

                # out[out >= 0.5] = 255
                # out[out < 0.5] = 0

                log_test.info('----------------------------------------------------------------------------------------------')
                log_test.info("loss -> " + str(loss))
                log_test.info(str(os.path.join(save_root, "{}_lab.png".format(root_name))))
                log_test.info(str(os.path.join(save_root, "{}_pre.png".format(root_name))))
                log_test.info('----------------------------------------------------------------------------------------------')
                cv2.imwrite(os.path.join(save_root, "{}_lab.png".format(root_name)), target)
                cv2.imwrite(os.path.join(save_root, "{}_pre.png".format(root_name)), out)
                pbar.set_description(f"Loss: {loss.item():.4f}")
                pbar.update(1)
        pbar.close()

        log_test.info("model -> " + str(epoch) + " test finish!")
        log_test.info('----------------------------------------------------------------------------------------------')
        print("testing epoch finish -> ", epoch)
        print("---------------------------------------------------------------------------------------")

        print("evalauting epoch start -> ", epoch)
        metrics = eval(log_eval, save_root, epoch)
        for key, value in metrics.items():
            print(str(key) + ' -> ' + str(value))
        if(max_mIoU < metrics['mIoU']):
            max_Metrics = metrics
            max_mIoU = metrics['mIoU']
            checkpoint_paths = [output_dir / f'checkpoint_best.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_train.info("\nupdate and save best model -> " + str(epoch))
            print("\nupdate and save best model -> ", epoch)

        print("evalauting epoch finish -> ", epoch)
        print('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        print("---------------------------------------------------------------------------------------")

        log_eval.info("evalauting epoch finish -> " + str(epoch))
        log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
        log_eval.info("---------------------------------------------------------------------------------------")

    if validation_loader is not None:
        if best_validation is None:
            raise RuntimeError('No validation-selected checkpoint was produced')
        print('Best validation metrics | ' + json.dumps(best_validation, allow_nan=False))
        log_train.info('Best validation metrics | ' + json.dumps(best_validation, allow_nan=False))
        completion = {'completed_epochs': args.epochs, 'selection': best_validation,
                      'checkpoint': str(output_dir / 'checkpoint_best_val.pth')}
        (output_dir / 'training_complete.json').write_text(json.dumps(completion, indent=2))
        print('Process time ' + str(datetime.timedelta(seconds=int(time.time() - start_time))))
        return

    # writer.close()

    for key, value in max_Metrics.items():
        log_eval.info(str(key) + ' -> ' + str(value))
    log_eval.info('\nmax_mIoU -> ' + str(max_Metrics['mIoU']) + '\nmax Epoch -> ' + str(max_Metrics['epoch']))
    best_summary = (
        f"Best metrics | experiment -> {experiment_name} | "
        f"epoch -> {max_Metrics['epoch']} | "
        f"mIoU -> {max_Metrics['mIoU']} | "
        f"F1 -> {max_Metrics['F1']} | "
        f"ODS -> {max_Metrics['ODS']} | "
        f"OIS -> {max_Metrics['OIS']} | "
        f"Precision -> {max_Metrics['Precision']} | "
        f"Recall -> {max_Metrics['Recall']}"
    )
    print(best_summary)
    log_train.info(best_summary)
    log_eval.info(best_summary)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Process time {}'.format(total_time_str))
    log_train.info('Process time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MixerCSeg FOR CRACK', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
