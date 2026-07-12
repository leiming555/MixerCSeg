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

    print('\n================ MixerCSeg Profile ================')
    print(f'Dataset      : {dataset_name}')
    print(f'Model        : {mode}')
    print(f'CCEM enabled : {args.use_ccem}')
    print(f'CCEM mode    : {args.ccem_mode if args.use_ccem else "baseline"}')
    print(f'Exp enabled  : {args.use_exp_module}')
    print(f'Exp mode     : {args.exp_module_mode}')
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
