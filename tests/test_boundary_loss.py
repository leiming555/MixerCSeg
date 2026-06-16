import torch
from types import SimpleNamespace

from losses import CompositeCrackLoss
from losses.boundary_loss import BoundaryLoss
from models.segmentor.MixerCSeg import bce_dice


def test_boundary_loss_is_finite_scalar_and_backward():
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss = BoundaryLoss()(torch.sigmoid(logits), target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_composite_boundary_loss_returns_dict_and_backward():
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss_out = CompositeCrackLoss(use_boundary=True)(logits, target)

    assert set(loss_out) == {
        "loss_total",
        "loss_bce",
        "loss_dice",
        "loss_boundary",
    }
    assert all(torch.isfinite(value) for value in loss_out.values())
    loss_out["loss_total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_target_is_finite():
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.zeros_like(logits)
    loss_out = CompositeCrackLoss(use_boundary=True)(logits, target)

    assert all(torch.isfinite(value) for value in loss_out.values())


def test_zero_logits_are_finite():
    logits = torch.zeros(2, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    loss_out = CompositeCrackLoss(use_boundary=True)(logits, target)

    assert all(torch.isfinite(value) for value in loss_out.values())
    loss_out["loss_total"].backward()
    assert torch.isfinite(logits.grad).all()


def test_boundary_disabled_uses_only_bce_and_dice():
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    loss_out = CompositeCrackLoss(use_boundary=False)(logits, target)
    expected = 0.87 * loss_out["loss_bce"] + 0.13 * loss_out["loss_dice"]

    assert loss_out["loss_boundary"].item() == 0.0
    assert torch.allclose(loss_out["loss_total"].detach(), expected)


def test_boundary_only_adds_weighted_boundary_term():
    logits = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, logits.shape).float()
    baseline = CompositeCrackLoss(use_boundary=False)(logits, target)
    boundary = CompositeCrackLoss(
        use_boundary=True,
        lambda_boundary=0.3,
    )(logits, target)
    expected = baseline["loss_total"] + 0.3 * boundary["loss_boundary"]

    assert torch.allclose(boundary["loss_total"], expected)


def test_boundary_disabled_matches_original_baseline_loss():
    logits = torch.randn(2, 1, 128, 128)
    target = torch.randint(0, 2, logits.shape).float()
    args = SimpleNamespace(BCELoss_ratio=0.87, DiceLoss_ratio=0.13)
    original_loss = bce_dice(args)(logits, target)
    composite_loss = CompositeCrackLoss(use_boundary=False)(logits, target)

    assert torch.allclose(composite_loss["loss_total"], original_loss)


def test_boundary_warmup_disables_boundary_term():
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randint(0, 2, logits.shape).float()
    criterion = CompositeCrackLoss(use_boundary=True, loss_warmup_epochs=5)
    loss_out = criterion(logits, target, epoch=2)

    assert loss_out["loss_boundary"].item() == 0.0
