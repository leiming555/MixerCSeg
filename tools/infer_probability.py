import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import create_dataset  # noqa: E402
from main import get_args_parser  # noqa: E402
from models import build_MixerCSeg as build_model  # noqa: E402


CHECKPOINT_ARG_NAMES = [
    "BCELoss_ratio",
    "DiceLoss_ratio",
    "use_tversky",
    "lambda_tversky",
    "tversky_alpha",
    "tversky_beta",
    "tversky_gamma",
    "tversky_multiscale",
    "tversky_tolerant_kernel",
    "pos_weight",
    "use_dilated_bce",
    "dilate_kernel",
    "use_boundary_loss",
    "lambda_boundary",
    "loss_warmup_epochs",
    "eps",
    "Norm_Type",
    "nbins",
    "use_ccem",
    "ccem_mode",
    "ccem_gate_mode",
    "ccem_branch_weight",
    "use_edrm",
    "edrm_stages",
    "use_exp_module",
    "exp_module_mode",
    "dataset_mode",
    "load_width",
    "load_height",
]


def get_parser():
    parser = argparse.ArgumentParser(
        "MixerCSeg probability-map inference",
        parents=[get_args_parser()],
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth file")
    parser.add_argument(
        "--save_dir",
        default="",
        help="Directory to save *_pre.png and *_lab.png files. Defaults to results/probability_maps/<checkpoint>_<dataset>_<phase>",
    )
    parser.add_argument(
        "--infer_phase",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split used for inference",
    )
    parser.add_argument(
        "--use_checkpoint_args",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse architecture/preprocessing args stored in the checkpoint when available",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict state_dict loading",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty save_dir",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle inference samples. Disabled by default for deterministic output order",
    )
    return parser


def resolve_device(device_arg):
    requested = torch.device(device_arg)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return requested


def load_checkpoint(path):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(str(checkpoint_path), map_location="cpu")


def apply_checkpoint_args(args, checkpoint):
    if not args.use_checkpoint_args:
        return args
    checkpoint_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    if checkpoint_args is None:
        return args
    for name in CHECKPOINT_ARG_NAMES:
        if hasattr(checkpoint_args, name):
            setattr(args, name, getattr(checkpoint_args, name))
    return args


def default_save_dir(args):
    checkpoint_name = Path(args.checkpoint).parent.name or Path(args.checkpoint).stem
    dataset_name = Path(args.dataset_path.rstrip("/")).name
    return PROJECT_ROOT / "results" / "probability_maps" / f"{checkpoint_name}_{dataset_name}_{args.infer_phase}"


def prepare_save_dir(args):
    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    existing_pngs = list(save_dir.glob("*.png"))
    if existing_pngs and not args.overwrite:
        raise FileExistsError(
            f"save_dir already contains PNG files: {save_dir}. "
            "Use --overwrite or choose a new --save_dir."
        )
    return save_dir


def state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def normalize_legacy_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        # Older CCEM checkpoints stored the gate as nn.Sequential(Conv2d, Sigmoid),
        # producing gate.0.* keys. The current module stores the Conv2d directly.
        new_key = key.replace(".gate.0.", ".gate.")
        normalized[new_key] = value
    return normalized


def image_stem(path):
    return Path(path).stem


@torch.no_grad()
def run_inference(args, model, data_loader, device, save_dir):
    model.eval()
    for data in tqdm(data_loader, total=len(data_loader), desc="Infer"):
        images = data["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)

        for batch_idx in range(images.shape[0]):
            root_name = image_stem(data["A_paths"][batch_idx])
            prob = probs[batch_idx, 0].detach().cpu().numpy()
            label = data["label"][batch_idx, 0].detach().cpu().numpy()

            pred_u8 = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
            label_u8 = ((label > 0.5).astype(np.uint8)) * 255

            cv2.imwrite(str(save_dir / f"{root_name}_pre.png"), pred_u8)
            cv2.imwrite(str(save_dir / f"{root_name}_lab.png"), label_u8)


def main():
    parser = get_parser()
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    args = apply_checkpoint_args(args, checkpoint)

    args.phase = args.infer_phase
    args.batch_size = args.batch_size_test
    args.serial_batches = not args.shuffle
    device = resolve_device(str(args.device))
    args.device = device

    save_dir = prepare_save_dir(args)
    data_loader = create_dataset(args)
    model, _ = build_model(args)
    state_dict = normalize_legacy_state_dict_keys(state_dict_from_checkpoint(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict)
    if missing or unexpected:
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")
    model.to(device)

    print(f"Dataset: {args.dataset_path}")
    print(f"Phase: {args.phase}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Save dir: {save_dir}")
    print(f"Samples: {len(data_loader)}")
    print("Output: sigmoid(logits) probability maps saved as *_pre.png")

    run_inference(args, model, data_loader, device, save_dir)
    print("Finished probability-map inference.")


if __name__ == "__main__":
    main()
