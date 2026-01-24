import pytest

torch = pytest.importorskip("torch")

from models.flows.conditional_spline_1d import ConditionalSplineFlow1D


def test_flow_diagnostics_keys():
    flow = ConditionalSplineFlow1D(context_dim=3, num_layers=2, num_bins=4)
    context = torch.randn(5, 3)
    r = 0.1 + 0.8 * torch.rand(5)

    log_prob, diagnostics = flow.log_prob(r, context, return_layer_logdet=True)

    assert log_prob.shape == r.shape
    assert set(diagnostics.keys()) == {
        "layer_logdet_mean_0",
        "layer_logdet_std_0",
        "layer_logdet_mean_1",
        "layer_logdet_std_1",
    }
