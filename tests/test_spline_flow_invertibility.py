import pytest


torch = pytest.importorskip("torch")

from tvae.models.flows.conditional_spline_1d import ConditionalSplineFlow1D


def test_spline_flow_invertibility():
    batch = 32
    context_dim = 8
    flow = ConditionalSplineFlow1D(
        context_dim=context_dim,
        num_layers=2,
        num_bins=6,
        min_bin_width=1e-2,
        min_bin_height=1e-2,
        min_derivative=1e-2,
        eps=1e-6,
    )

    context = torch.randn(batch, context_dim)
    z = torch.rand(batch) * 0.98 + 0.01

    x, logdet_fwd = flow.forward(z, context)
    z_hat, logdet_inv = flow.inverse(x, context)

    assert torch.max(torch.abs(z - z_hat)).item() < 1e-4
    assert torch.max(torch.abs(logdet_fwd + logdet_inv)).item() < 1e-4
