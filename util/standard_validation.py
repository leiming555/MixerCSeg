"""Validation-only checkpoint selection, matching the PNG evaluation protocol."""

import copy
import math

import numpy as np
import torch

from datasets import create_dataset
from tools.eval_standard import metrics_from_counts


def quantize_probability(logits):
    probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    if not np.isfinite(probabilities).all() or not torch.isfinite(logits).all():
        raise ValueError("Non-finite validation predictions")
    return np.clip(np.rint(probabilities * 255.0), 0, 255).astype(np.uint8)


def make_validation_loader(args):
    val_args = copy.copy(args)
    val_args.phase = "val"
    val_args.batch_size = args.batch_size_test
    val_args.serial_batches = True
    return create_dataset(val_args)


def selection_key(metrics, epoch):
    values = (float(metrics["mIoU"]), float(metrics["F1"]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite selection metric")
    return values + (-int(epoch),)


@torch.no_grad()
def evaluate_validation(model, loader, device):
    if getattr(loader.dataset, "phase", None) != "val":
        raise ValueError("Checkpoint selection must use the val split")
    counts = np.zeros(4, dtype=np.int64)
    seen = 0
    was_training = model.training
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        model.eval()
        # Evaluation must not change the next epoch's training shuffle RNG.
        with torch.random.fork_rng(devices=devices):
            for batch in loader:
                logits = model(batch["image"].to(device))
                predictions = quantize_probability(logits) > 127
                labels = batch["label"].cpu().numpy() > 0.5
                if predictions.shape != labels.shape:
                    raise ValueError("Validation prediction/label shape mismatch")
                counts += [np.sum(predictions & labels), np.sum(predictions & ~labels),
                           np.sum(~predictions & labels), np.sum(~predictions & ~labels)]
                seen += len(predictions)
    finally:
        model.train(was_training)
    if seen != len(loader.dataset) or seen == 0:
        raise ValueError("Incomplete or empty validation split")
    return metrics_from_counts(*(int(value) for value in counts))
