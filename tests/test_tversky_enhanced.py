import torch

from losses.base_losses import TverskyLoss


def test_tversky_default_matches_original_formula():
    prob = torch.rand(2, 1, 16, 16)
    target = torch.randint(0, 2, prob.shape).float()
    loss = TverskyLoss(alpha=0.3, beta=0.7, eps=1e-6)(prob, target)

    dims = tuple(range(2, prob.ndim))
    tp = (prob * target).sum(dims)
    fp = (prob * (1.0 - target)).sum(dims)
    fn = ((1.0 - prob) * target).sum(dims)
    score = (tp + 1e-6) / (tp + 0.3 * fp + 0.7 * fn + 1e-6)
    expected = 1.0 - score.mean()

    assert torch.allclose(loss, expected)


def test_enhanced_tversky_is_finite_and_backward():
    logits = torch.randn(2, 1, 31, 29, requires_grad=True)
    prob = torch.sigmoid(logits)
    target = torch.randint(0, 2, prob.shape).float()
    loss = TverskyLoss(
        alpha=0.3,
        beta=0.7,
        gamma=1.33,
        multiscale=True,
        tolerant_kernel=3,
    )(prob, target)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_enhanced_tversky_zero_target_is_finite():
    prob = torch.rand(2, 1, 16, 16)
    target = torch.zeros_like(prob)
    loss = TverskyLoss(gamma=1.33, multiscale=True, tolerant_kernel=3)(prob, target)

    assert torch.isfinite(loss)
