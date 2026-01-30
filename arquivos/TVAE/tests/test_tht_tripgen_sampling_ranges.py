import pytest


torch = pytest.importorskip("torch")

from models.tht_tripgen import THTTripGenModel


def test_tht_tripgen_sampling_ranges():
    num_zones = 6
    num_time_bins = 3
    emb_dim = 8
    cond_card = {"dow_idx": 4}

    zone_emb = torch.randn(num_zones, emb_dim)
    model = THTTripGenModel(
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

    sample_a = model.sample(100, seed=123)
    sample_b = model.sample(100, seed=123)

    assert torch.equal(sample_a["h_idx"], sample_b["h_idx"])
    assert torch.equal(sample_a["o_idx"], sample_b["o_idx"])
    assert torch.equal(sample_a["d_idx"], sample_b["d_idx"])
    assert torch.equal(sample_a["dow_idx"], sample_b["dow_idx"])

    assert sample_a["h_idx"].min() >= 0
    assert sample_a["h_idx"].max() < num_time_bins
    assert sample_a["o_idx"].min() >= 0
    assert sample_a["o_idx"].max() < num_zones
    assert sample_a["d_idx"].min() >= 0
    assert sample_a["d_idx"].max() < num_zones
    assert sample_a["dow_idx"].min() >= 0
    assert sample_a["dow_idx"].max() < cond_card["dow_idx"]

    assert torch.all(sample_a["r"] > 0)
    assert torch.all(sample_a["r"] < 1)
