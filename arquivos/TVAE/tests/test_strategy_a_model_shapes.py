import pytest


torch = pytest.importorskip("torch")

from models.strategy_a import StrategyAModel


def test_strategy_a_model_shapes():
    num_zones = 5
    num_time_bins = 4
    emb_dim = 8
    cond_card = {"dow_idx": 3}

    zone_emb = torch.randn(num_zones, emb_dim)
    model = StrategyAModel(
        num_zones=num_zones,
        num_time_bins=num_time_bins,
        conditional_cardinalities=cond_card,
        frozen_zone_embeddings=zone_emb,
        cond_emb_dim=4,
        time_emb_dim=4,
        origin_emb_dim=4,
        context_mlp_hidden=16,
        context_mlp_layers=2,
        dropout=0.0,
    )

    batch = 6
    u = {"dow_idx": torch.randint(0, cond_card["dow_idx"], (batch,))}
    h_idx = torch.randint(0, num_time_bins, (batch,))
    o_idx = torch.randint(0, num_zones, (batch,))
    d_idx = torch.randint(0, num_zones, (batch,))
    r = torch.rand(batch)

    logits = model(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx)

    assert logits["logits_h"].shape == (batch, num_time_bins)
    assert logits["logits_o"].shape == (batch, num_zones)
    assert logits["logits_d"].shape == (batch, num_zones)

    nll = model.nll(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx, r=r)
    assert torch.isfinite(nll["nll_total"]).item()
