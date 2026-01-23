import pytest


torch = pytest.importorskip("torch")

from models.flows.conditional_spline_1d import ConditionalSplineFlow1D


def test_residual_flow_sampling_range():
    flow = ConditionalSplineFlow1D(
        context_dim=4,
        num_layers=2,
        num_bins=5,
        min_bin_width=1e-2,
        min_bin_height=1e-2,
        min_derivative=1e-2,
        eps=1e-6,
    )

    context = torch.randn(64, 4)
    samples = flow.sample(context, seed=123)

    assert torch.all(torch.isfinite(samples))
    assert torch.all(samples > 0)
    assert torch.all(samples < 1)
