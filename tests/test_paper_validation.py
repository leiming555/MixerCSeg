import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from main import get_args_parser
from tools.eval_standard import eval_threshold
from tools.paper_audit import audit_dataset, completed, core_jobs, REMOVALS, validate_checkpoint, family_name
from util.standard_validation import (
    evaluate_validation, make_validation_loader, quantize_probability, selection_key,
)


class Loader:
    def __init__(self, batch, split="val"):
        self.dataset = type("Dataset", (), {"phase": split, "__len__": lambda _: len(batch["image"])})()
        self.batch = batch

    def __iter__(self):
        torch.rand(1)
        yield self.batch


def test_quantization_matches_png_protocol():
    logits = torch.tensor([[[[-30., -0.01, 0., 0.01, 30.]]]])
    expected = np.clip(np.rint(torch.sigmoid(logits).numpy() * 255), 0, 255).astype(np.uint8)
    assert np.array_equal(quantize_probability(logits), expected)
    assert expected[0, 0, 0, 2] == 128


def test_standard_validation_matches_offline_counts_and_preserves_rng():
    logits = torch.tensor([[[[-3., 0., 3.], [-1., 1., 4.]]]])
    labels = torch.tensor([[[[0., 0., 1.], [1., 1., 0.]]]])
    loader = Loader({"image": logits, "label": labels})
    model = torch.nn.Identity().train()
    rng = torch.get_rng_state().clone()
    actual = evaluate_validation(model, loader, "cpu")
    assert torch.equal(rng, torch.get_rng_state())
    assert model.training
    pairs = [(quantize_probability(logits)[0, 0].astype(np.float32) / 255,
              labels.numpy()[0, 0].astype(np.uint8))]
    assert actual == eval_threshold(pairs, 0.5)


def test_selection_rejects_test_loader_and_nonfinite_predictions():
    batch = {"image": torch.zeros(1, 1, 2, 2), "label": torch.zeros(1, 1, 2, 2)}
    with pytest.raises(ValueError, match="val split"):
        evaluate_validation(torch.nn.Identity(), Loader(batch, "test"), "cpu")
    batch["image"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        evaluate_validation(torch.nn.Identity(), Loader(batch), "cpu")


def test_selection_ties_choose_f1_then_earliest_epoch():
    a = {"mIoU": 0.8, "F1": 0.7}
    b = {"mIoU": 0.8, "F1": 0.71}
    assert selection_key(b, 20) > selection_key(a, 1)
    assert selection_key(a, 1) > selection_key(a, 2)
    with pytest.raises(ValueError):
        selection_key({"mIoU": float("nan"), "F1": 0.5}, 0)


def test_validation_loader_copies_args_without_touching_train(monkeypatch):
    args = SimpleNamespace(phase="train", batch_size=3, batch_size_test=1, serial_batches=False)
    from util import standard_validation
    monkeypatch.setattr(standard_validation, "create_dataset", lambda options: options)
    val = make_validation_loader(args)
    assert val.phase == "val" and val.serial_batches and val.batch_size == 1
    assert args.phase == "train" and args.batch_size == 3 and not args.serial_batches


def test_validation_flag_is_opt_in_and_loop_bypasses_legacy_test():
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    assert not parser.parse_args([]).validation_selection
    assert parser.parse_args(["--validation_selection"]).validation_selection
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    selection = source.index("if validation_loader is not None:", source.index("for epoch in range"))
    legacy = source.index("args.phase = 'test'", selection)
    assert "continue" in source[selection:legacy]
    assert "checkpoint_best_val.pth" in source[selection:legacy]


def make_split(root, split, value):
    (root / f"{split}_img").mkdir(parents=True)
    (root / f"{split}_lab").mkdir()
    cv2.imwrite(str(root / f"{split}_img/sample.png"), np.full((4, 4, 3), value, np.uint8))
    cv2.imwrite(str(root / f"{split}_lab/sample.png"), np.zeros((4, 4), np.uint8))


def test_dataset_audit_accepts_same_empty_masks_but_rejects_duplicate_images(tmp_path):
    for split, value in [("train", 10), ("val", 20), ("test", 30)]:
        make_split(tmp_path, split, value)
    assert audit_dataset(tmp_path)["errors"] == []
    cv2.imwrite(str(tmp_path / "test_img/sample.png"), np.full((4, 4, 3), 10, np.uint8))
    assert any("Cross-split duplicate" in e for e in audit_dataset(tmp_path)["errors"])


def test_dataset_audit_rejects_missing_or_unpaired_labels(tmp_path):
    for i, split in enumerate(("train", "val", "test")):
        make_split(tmp_path, split, i)
    (tmp_path / "val_lab/sample.png").unlink()
    assert any("Unreadable pair" in e for e in audit_dataset(tmp_path)["errors"])


def test_completion_requires_marker_metrics_and_artifacts(tmp_path):
    (tmp_path / "checkpoint_best_val.pth").touch()
    assert not completed(tmp_path)
    record = {"checkpoint": str(tmp_path / "checkpoint_best_val.pth"),
              "result_dir": str(tmp_path), "selection_split": "val"}
    (tmp_path / "completed.json").write_text(json.dumps(record))
    assert not completed(tmp_path)


def test_manifest_has_18_unique_core_jobs_and_four_removals():
    jobs = core_jobs()
    assert len(jobs) == len({job["job_id"] for job in jobs}) == 18
    assert len([j for j in jobs if j["dataset"] == "CrackMap"]) == 12
    assert len(REMOVALS) == 4


def test_checkpoint_protocol_rejects_extra_loss():
    args = SimpleNamespace(epochs=50, seed=42, nbins=180, dataset_path="dataset/CrackMap",
                           BCELoss_ratio=.87, DiceLoss_ratio=.13, use_tversky=True)
    job = dict(job_id="baseline", dataset="CrackMap", model="baseline", seed=42)
    with pytest.raises(ValueError, match="Unexpected active flags"):
        validate_checkpoint({"args": args}, job)


def test_family_uses_experiment_name_not_batch_directory():
    parent = Path("work_dirs/triple_stack_v4/CrackMap_seed42_ccem_baseline/checkpoint_best.pth").parent
    assert family_name(parent.name) == "baseline"
    assert family_name("CrackMap_seed42_ccem_baseline_triple_stack_v3_msa_ocv_gbc") == "triple_stack_v3"


def test_training_validation_branch_never_opens_test_split(tmp_path, monkeypatch):
    import main as training
    from util import standard_validation

    cwd = tmp_path / "project"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    accessed = []
    batch = {"image": torch.ones(1, 1, 2, 2), "label": torch.ones(1, 1, 2, 2)}

    def dataset(args):
        accessed.append(args.phase)
        assert args.phase in ("train", "val")
        loader = Loader(batch, args.phase)
        loader.__class__.__len__ = lambda self: 1
        return loader

    monkeypatch.setattr(training, "create_dataset", dataset)
    monkeypatch.setattr(standard_validation, "create_dataset", dataset)
    monkeypatch.setattr(training, "build_model",
                        lambda _: (torch.nn.Conv2d(1, 1, 1), torch.nn.BCEWithLogitsLoss()))
    monkeypatch.setattr(training, "get_logger", lambda *_: SimpleNamespace(info=lambda _: None))

    def train(model, criterion, loader, optimizer, *unused):
        model.train()
        optimizer.zero_grad()
        criterion(model(batch["image"]), batch["label"]).backward()
        optimizer.step()

    monkeypatch.setattr(training, "train_one_epoch", train)
    args = get_args_parser().parse_args([
        "--device", "cpu", "--epochs", "2", "--lr_scheduler", "StepLR",
        "--dataset_path", "dataset/CrackMap", "--output_dir", str(tmp_path / "weights"),
        "--validation_selection",
    ])
    training.main(args)
    assert accessed == ["train", "val"]
    best = list((tmp_path / "weights").glob("*/checkpoint_best_val.pth"))
    assert len(best) == 1
    checkpoint = torch.load(best[0], map_location="cpu")
    assert checkpoint["selection"]["split"] == "val"
    assert not list((tmp_path / "weights").glob("*/checkpoint_best.pth"))
    history = (best[0].parent / "validation_history.jsonl").read_text().splitlines()
    assert len(history) == 2
    assert json.loads((best[0].parent / "training_complete.json").read_text())["completed_epochs"] == 2
