import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_MixerCSeg as build_model  # noqa: E402
from models.experimental_modules import EXPERIMENTAL_MODULE_MODES  # noqa: E402
from models.experimental_second_modules import SECOND_EXPERIMENTAL_MODULE_MODES  # noqa: E402
from models.experimental_third_modules import THIRD_EXPERIMENTAL_MODULE_MODES  # noqa: E402
from models.experimental_paper_stack import PAPER_STACK_MODES  # noqa: E402
from models.deg_ridge_scale_gap_interaction import DRSGI_MODES  # noqa: E402
from models.rsgdi_v2_modules import RSGDI_V2_MODES  # noqa: E402
from models.experimental_triple_stack_v3 import TRIPLE_STACK_V3_MODES  # noqa: E402
from models.experimental_triple_stack_v4 import TRIPLE_STACK_V4_MODES  # noqa: E402
from models.experimental_triple_stack_v5 import TRIPLE_STACK_V5_MODES  # noqa: E402
from models.experimental_triple_stack_v6 import TRIPLE_STACK_V6_MODES  # noqa: E402
from models.experimental_triple_stack_v7 import TRIPLE_STACK_V7_MODES  # noqa: E402
from models.experimental_triple_stack_v8 import TRIPLE_STACK_V8_MODES  # noqa: E402
from models.experimental_triple_stack_v9 import TRIPLE_STACK_V9_MODES  # noqa: E402
from models.experimental_triple_stack_v10 import TRIPLE_STACK_V10_MODES  # noqa: E402
from models.experimental_triple_stack_v11 import TRIPLE_STACK_V11_MODES  # noqa: E402
from models.experimental_triple_stack_v12 import TRIPLE_STACK_V12_MODES  # noqa: E402


def get_args():
    parser = argparse.ArgumentParser('Profile MixerCSeg complexity')
    parser.add_argument('--dataset_path', default='dataset/CrackMap',
                        help='Dataset path, only used for experiment naming consistency')
    parser.add_argument('--nbins', default=180, type=int,
                        help='Number of direction intervals used by DEGConv')
    parser.add_argument('--use_ccem', action='store_true',
                        help='Enable CCEM after DEGConv in every encoder stage')
    parser.add_argument('--ccem_mode', default='full', type=str,
                        choices=['full', 'enhanced', 'transformer', 'no_local', 'no_strip', 'no_dilation', 'no_gate'],
                        help='CCEM ablation mode')
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
                        help='Enable RSGDI-v2 after DEGConv')
    parser.add_argument('--rsgdi_v2_mode', default='topology_bridge_gate', type=str,
                        choices=RSGDI_V2_MODES,
                        help='RSGDI-v2 enhancement mode')
    parser.add_argument('--use_triple_stack_v3', action='store_true',
                        help='Enable TripleStack-v3 with three fixed innovation points around DEGConv')
    parser.add_argument('--triple_stack_v3_mode', default='dsc_htm_gbc', type=str,
                        choices=TRIPLE_STACK_V3_MODES,
                        help='TripleStack-v3 mode')
    parser.add_argument('--use_triple_stack_v4', action='store_true',
                        help='Enable TripleStack-v4 around DEGConv')
    parser.add_argument('--triple_stack_v4_mode', default='amc_eov_tgc', type=str,
                        choices=TRIPLE_STACK_V4_MODES,
                        help='TripleStack-v4 mode')
    parser.add_argument('--use_triple_stack_v5', action='store_true',
                        help='Enable TripleStack-v5 around DEGConv')
    parser.add_argument('--triple_stack_v5_mode', default='mwr_eoc_upb', type=str,
                        choices=TRIPLE_STACK_V5_MODES,
                        help='TripleStack-v5 mode')
    parser.add_argument('--use_triple_stack_v6', action='store_true',
                        help='Enable TripleStack-v6 around DEGConv')
    parser.add_argument('--triple_stack_v6_mode', default='hfr_ert_apb', type=str,
                        choices=TRIPLE_STACK_V6_MODES,
                        help='TripleStack-v6 mode')
    parser.add_argument('--use_triple_stack_v7', action='store_true',
                        help='Enable TripleStack-v7 around DEGConv')
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
    parser.add_argument('--use_triple_stack_v12', action='store_true',
                        help='Enable parallel-evidence TripleStack-v12 around DEGConv')
    parser.add_argument('--triple_stack_v12_mode', default='ase_cpa_pgg', type=str,
                        choices=TRIPLE_STACK_V12_MODES,
                        help='TripleStack-v12 mode')
    parser.add_argument('--input_size', default=512, type=int,
                        help='Input image size, using square input')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Dummy input batch size')
    parser.add_argument('--warmup', default=50, type=int,
                        help='Warmup iterations before timing')
    parser.add_argument('--repeat', default=300, type=int,
                        help='Measured iterations for FPS and latency')
    return parser.parse_args()


def build_profile_args(args, device):
    return SimpleNamespace(
        BCELoss_ratio=0.87,
        DiceLoss_ratio=0.13,
        Norm_Type='GN',
        nbins=args.nbins,
        use_ccem=args.use_ccem,
        ccem_mode=args.ccem_mode,
        use_exp_module=args.use_exp_module,
        exp_module_mode=args.exp_module_mode,
        use_exp_second_module=args.use_exp_second_module,
        exp_second_module_mode=args.exp_second_module_mode,
        use_exp_third_module=args.use_exp_third_module,
        exp_third_module_mode=args.exp_third_module_mode,
        use_paper_stack=args.use_paper_stack,
        paper_stack_mode=args.paper_stack_mode,
        use_drsgi=args.use_drsgi,
        drsgi_mode=args.drsgi_mode,
        use_rsgdi_v2=args.use_rsgdi_v2,
        rsgdi_v2_mode=args.rsgdi_v2_mode,
        use_triple_stack_v3=args.use_triple_stack_v3,
        triple_stack_v3_mode=args.triple_stack_v3_mode,
        use_triple_stack_v4=args.use_triple_stack_v4,
        triple_stack_v4_mode=args.triple_stack_v4_mode,
        use_triple_stack_v5=args.use_triple_stack_v5,
        triple_stack_v5_mode=args.triple_stack_v5_mode,
        use_triple_stack_v6=args.use_triple_stack_v6,
        triple_stack_v6_mode=args.triple_stack_v6_mode,
        use_triple_stack_v7=args.use_triple_stack_v7,
        triple_stack_v7_mode=args.triple_stack_v7_mode,
        use_triple_stack_v8=args.use_triple_stack_v8,
        triple_stack_v8_mode=args.triple_stack_v8_mode,
        use_triple_stack_v9=args.use_triple_stack_v9,
        triple_stack_v9_mode=args.triple_stack_v9_mode,
        use_triple_stack_v10=args.use_triple_stack_v10,
        triple_stack_v10_mode=args.triple_stack_v10_mode,
        use_triple_stack_v11=args.use_triple_stack_v11,
        triple_stack_v11_mode=args.triple_stack_v11_mode,
        use_triple_stack_v12=args.use_triple_stack_v12,
        triple_stack_v12_mode=args.triple_stack_v12_mode,
        dataset_path=args.dataset_path,
        device=device,
        load_width=args.input_size,
        load_height=args.input_size,
    )


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def profile_flops(model, dummy_input):
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            'thop is required for FLOPs profiling. Install it with: pip install thop'
        ) from exc

    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    return flops, params


@torch.no_grad()
def benchmark(model, dummy_input, warmup, repeat, device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model.eval()

    for _ in range(warmup):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(repeat):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    latency_ms = elapsed / repeat * 1000.0
    fps = dummy_input.shape[0] * repeat / elapsed
    peak_memory_mb = 0.0
    if device.type == 'cuda':
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return fps, latency_ms, peak_memory_mb


def print_results(args, params_m, flops_g, size_mb, fps, latency_ms, peak_memory_mb):
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    mode = args.ccem_mode if args.use_ccem else 'baseline'
    if args.use_exp_module:
        mode = f'{mode}_exp_{args.exp_module_mode}'
    if args.use_exp_second_module:
        mode = f'{mode}_second_{args.exp_second_module_mode}'
    if args.use_exp_third_module:
        mode = f'{mode}_third_{args.exp_third_module_mode}'
    if args.use_paper_stack:
        mode = f'{mode}_paper_{args.paper_stack_mode}'
    if args.use_drsgi:
        mode = f'{mode}_drsgi_{args.drsgi_mode}'
    if args.use_rsgdi_v2:
        mode = f'{mode}_rsgdi_v2_{args.rsgdi_v2_mode}'
    if args.use_triple_stack_v3:
        mode = f'{mode}_triple_stack_v3_{args.triple_stack_v3_mode}'
    if args.use_triple_stack_v4:
        mode = f'{mode}_triple_stack_v4_{args.triple_stack_v4_mode}'
    if args.use_triple_stack_v5:
        mode = f'{mode}_triple_stack_v5_{args.triple_stack_v5_mode}'
    if args.use_triple_stack_v6:
        mode = f'{mode}_triple_stack_v6_{args.triple_stack_v6_mode}'
    if args.use_triple_stack_v7:
        mode = f'{mode}_triple_stack_v7_{args.triple_stack_v7_mode}'
    if args.use_triple_stack_v8:
        mode = f'{mode}_triple_stack_v8_{args.triple_stack_v8_mode}'
    if args.use_triple_stack_v9:
        mode = f'{mode}_triple_stack_v9_{args.triple_stack_v9_mode}'
    if args.use_triple_stack_v10:
        mode = f'{mode}_triple_stack_v10_{args.triple_stack_v10_mode}'
    if args.use_triple_stack_v11:
        mode = f'{mode}_triple_stack_v11_{args.triple_stack_v11_mode}'
    if args.use_triple_stack_v12:
        mode = f'{mode}_triple_stack_v12_{args.triple_stack_v12_mode}'

    print('\n================ MixerCSeg Profile ================')
    print(f'Dataset      : {dataset_name}')
    print(f'Model        : {mode}')
    print(f'CCEM enabled : {args.use_ccem}')
    print(f'CCEM mode    : {args.ccem_mode if args.use_ccem else "baseline"}')
    print(f'Exp enabled  : {args.use_exp_module}')
    print(f'Exp mode     : {args.exp_module_mode}')
    print(f'Second exp enabled: {args.use_exp_second_module}')
    print(f'Second exp mode   : {args.exp_second_module_mode}')
    print(f'Third exp enabled : {args.use_exp_third_module}')
    print(f'Third exp mode    : {args.exp_third_module_mode}')
    print(f'Paper stack enabled: {args.use_paper_stack}')
    print(f'Paper stack mode   : {args.paper_stack_mode}')
    print(f'DRSGI enabled      : {args.use_drsgi}')
    print(f'DRSGI mode         : {args.drsgi_mode}')
    print(f'RSGDI-v2 enabled  : {args.use_rsgdi_v2}')
    print(f'RSGDI-v2 mode     : {args.rsgdi_v2_mode}')
    print(f'TripleStack-v3 enabled: {args.use_triple_stack_v3}')
    print(f'TripleStack-v3 mode   : {args.triple_stack_v3_mode}')
    print(f'TripleStack-v4 enabled: {args.use_triple_stack_v4}')
    print(f'TripleStack-v4 mode   : {args.triple_stack_v4_mode}')
    print(f'TripleStack-v5 enabled: {args.use_triple_stack_v5}')
    print(f'TripleStack-v5 mode   : {args.triple_stack_v5_mode}')
    print(f'TripleStack-v6 enabled: {args.use_triple_stack_v6}')
    print(f'TripleStack-v6 mode   : {args.triple_stack_v6_mode}')
    print(f'TripleStack-v7 enabled: {args.use_triple_stack_v7}')
    print(f'TripleStack-v7 mode   : {args.triple_stack_v7_mode}')
    print(f'TripleStack-v8 enabled: {args.use_triple_stack_v8}')
    print(f'TripleStack-v8 mode   : {args.triple_stack_v8_mode}')
    print(f'TripleStack-v9 enabled: {args.use_triple_stack_v9}')
    print(f'TripleStack-v9 mode   : {args.triple_stack_v9_mode}')
    print(f'TripleStack-v10 enabled: {args.use_triple_stack_v10}')
    print(f'TripleStack-v10 mode   : {args.triple_stack_v10_mode}')
    print(f'TripleStack-v11 enabled: {args.use_triple_stack_v11}')
    print(f'TripleStack-v11 mode   : {args.triple_stack_v11_mode}')
    print(f'TripleStack-v12 enabled: {args.use_triple_stack_v12}')
    print(f'TripleStack-v12 mode   : {args.triple_stack_v12_mode}')
    print(f'Input        : {args.batch_size} x 3 x {args.input_size} x {args.input_size}')
    print(f'NBINS        : {args.nbins}')
    print('---------------------------------------------------')
    print(f'Params (M)        : {params_m:.3f}')
    print(f'FLOPs (GFLOPs)    : {flops_g:.3f}')
    print(f'Model Size (MB)   : {size_mb:.3f}')
    print(f'FPS               : {fps:.3f}')
    print(f'Latency (ms/img)  : {latency_ms / args.batch_size:.3f}')
    print(f'Latency (ms/batch): {latency_ms:.3f}')
    print(f'GPU Memory (MB)   : {peak_memory_mb:.3f}')
    print('---------------------------------------------------')
    print('Paper row:')
    print('Model\tParams(M)\tFLOPs(G)\tSize(MB)\tFPS\tLatency(ms/img)\tGPU Mem(MB)')
    print(
        f'{mode}\t{params_m:.3f}\t{flops_g:.3f}\t{size_mb:.3f}\t'
        f'{fps:.3f}\t{latency_ms / args.batch_size:.3f}\t{peak_memory_mb:.3f}'
    )
    print('===================================================\n')


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    profile_args = build_profile_args(args, device)

    model, _ = build_model(profile_args)
    model.to(device)
    model.eval()

    dummy_input = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=device,
    )

    params = count_params(model)
    size_mb = model_size_mb(model)
    try:
        flops, _ = profile_flops(model, dummy_input)
    except ImportError as exc:
        print(str(exc))
        sys.exit(1)
    fps, latency_ms, peak_memory_mb = benchmark(
        model,
        dummy_input,
        args.warmup,
        args.repeat,
        device,
    )

    print_results(
        args=args,
        params_m=params / 1e6,
        flops_g=flops / 1e9,
        size_mb=size_mb,
        fps=fps,
        latency_ms=latency_ms,
        peak_memory_mb=peak_memory_mb,
    )


if __name__ == '__main__':
    main()
