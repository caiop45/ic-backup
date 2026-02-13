import pytest


torch = pytest.importorskip("torch")

from tvae.models.tht_tripgen import THTTripGenModel


def test_tht_tripgen_nll_includes_r():
    num_zones = 4
    num_time_bins = 3
    cond_card = {"dow_idx": 2}
    zone_emb = torch.randn(num_zones, 6)

    model = THTTripGenModel(
        num_zones=num_zones,
        num_time_bins=num_time_bins,
        conditional_cardinalities=cond_card,
        frozen_zone_embeddings=zone_emb,
        cond_emb_dim=4,
        time_emb_dim=4,
        origin_emb_dim=4,
        context_mlp_hidden=8,
        context_mlp_layers=2,
        dropout=0.0,
        use_passenger_count=False,
        passenger_cardinality=None,
        use_total_amount=False,
        total_amount_sigma_floor=1e-4,
        residual_num_layers=2,
        residual_num_bins=5,
        residual_context_hidden=8,
        residual_min_bin_width=1e-2,
        residual_min_deriv=1e-2,
    )

    batch = 8
    u = {"dow_idx": torch.randint(0, cond_card["dow_idx"], (batch,))}
    h_idx = torch.randint(0, num_time_bins, (batch,))
    o_idx = torch.randint(0, num_zones, (batch,))
    d_idx = torch.randint(0, num_zones, (batch,))
    r = torch.rand(batch) * 0.98 + 0.01

    nll = model.nll(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx, r=r)
    assert "nll_r" in nll
    assert torch.isfinite(nll["nll_r"]).item()
    assert torch.isfinite(nll["nll_total"]).item()
