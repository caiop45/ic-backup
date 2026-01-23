import pytest


torch = pytest.importorskip("torch")

from models.flows.conditional_spline_1d import ConditionalSplineFlow1D


def test_conditional_spline_log_prob_handles_boundaries():
    eps = 1e-6
    flow = ConditionalSplineFlow1D(
        context_dim=3,
        num_layers=2,
        num_bins=5,
        min_bin_width=1e-2,
        min_bin_height=1e-2,
        min_derivative=1e-2,
        eps=eps,
    )

    r = torch.tensor([eps, 1.0 - eps], dtype=torch.float32)
    context = torch.randn(2, 3)
    log_prob = flow.log_prob(r, context)

    assert torch.all(torch.isfinite(log_prob))
