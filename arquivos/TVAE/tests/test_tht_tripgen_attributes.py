import numpy as np
import pandas as pd
import pytest

import config
from data_processing.tht_tripgen_transformer import THTTripGenTransformer


def test_transformer_amount_roundtrip(monkeypatch):
    monkeypatch.setattr(config, "THT_TOTAL_AMOUNT_CLIP_PMIN", 0.0)
    monkeypatch.setattr(config, "THT_TOTAL_AMOUNT_CLIP_PMAX", 1.0)

    df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2, 3],
            "r": [0.1, 0.2, 0.3, 0.4],
            "pickup_id": [1, 2, 3, 4],
            "dropoff_id": [2, 3, 4, 1],
            "dia_da_semana": [0, 1, 2, 3],
            "passenger_count": [1, 2, 1, 3],
            "total_amount": [10.0, 20.0, 30.0, 40.0],
        }
    )

    transformer = THTTripGenTransformer().fit(df)
    encoded = transformer.transform(df)
    decoded = transformer.decode(encoded)

    assert "total_amount" in decoded.columns
    assert np.allclose(
        decoded["total_amount"].to_numpy(dtype=np.float64),
        df["total_amount"].to_numpy(dtype=np.float64),
        rtol=1e-3,
        atol=1e-3,
    )


def test_model_outputs_include_attributes_when_enabled():
    torch = pytest.importorskip("torch")
    from models.tht_tripgen import THTTripGenModel

    num_zones = 5
    num_time_bins = 4
    emb_dim = 6
    cond_card = {"dow_idx": 3}
    zone_embeddings = torch.randn(num_zones, emb_dim)

    model = THTTripGenModel(
        num_zones=num_zones,
        num_time_bins=num_time_bins,
        conditional_cardinalities=cond_card,
        frozen_zone_embeddings=zone_embeddings,
        cond_emb_dim=4,
        time_emb_dim=4,
        origin_emb_dim=5,
        context_mlp_hidden=12,
        context_mlp_layers=2,
        dropout=0.0,
        destination_head_type="embedding_softmax",
        use_passenger_count=True,
        passenger_cardinality=4,
        use_total_amount=True,
        total_amount_sigma_floor=1e-4,
        min_r_eps=1e-6,
        residual_num_layers=2,
        residual_num_bins=4,
        residual_context_hidden=8,
        residual_min_bin_width=1e-2,
        residual_min_bin_height=1e-2,
        residual_min_deriv=1e-2,
        residual_eps=1e-6,
    )

    sample = model.sample(8, seed=123)
    assert "passenger_idx" in sample
    assert "total_amount_z" in sample
    assert sample["passenger_idx"].shape == (8,)
    assert sample["total_amount_z"].shape == (8,)
