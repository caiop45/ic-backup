import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.privacy import exact_match_rate, exact_match_rate_with_r


def test_exact_match_rate_basic():
    train_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1],
            "hora_do_dia": [5, 6],
            "pickup_id": [10, 11],
            "dropoff_id": [20, 21],
        }
    )
    synth_same = train_df.copy()
    synth_diff = pd.DataFrame(
        {
            "dia_da_semana": [2, 3],
            "hora_do_dia": [7, 8],
            "pickup_id": [12, 13],
            "dropoff_id": [22, 23],
        }
    )

    cols = ["dia_da_semana", "hora_do_dia", "pickup_id", "dropoff_id"]
    assert exact_match_rate(train_df, synth_same, cols) == 1.0
    assert exact_match_rate(train_df, synth_diff, cols) == 0.0


def test_exact_match_rate_with_r_rounding():
    train_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1],
            "hora_do_dia": [5, 6],
            "pickup_id": [10, 11],
            "dropoff_id": [20, 21],
            "r": [0.1234, 0.5678],
        }
    )
    synth_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1],
            "hora_do_dia": [5, 6],
            "pickup_id": [10, 11],
            "dropoff_id": [20, 21],
            "r": [0.1249, 0.5601],
        }
    )

    cols = ["dia_da_semana", "hora_do_dia", "pickup_id", "dropoff_id"]
    rate_2 = exact_match_rate_with_r(
        train_df, synth_df, cols, r_col="r", decimals=2
    )
    rate_3 = exact_match_rate_with_r(
        train_df, synth_df, cols, r_col="r", decimals=3
    )

    assert rate_2 == 1.0
    assert rate_3 == 0.0


def test_sample_tht_tripgen_privacy_report(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    import sample_tht_tripgen
    from data_processing.tht_tripgen_transformer import THTTripGenTransformer
    from models.tht_tripgen import THTTripGenModel
    from utils.serialization import save_checkpoint, save_tht_tripgen_mappings

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
    train_df = df.iloc[:3].reset_index(drop=True)
    val_df = df.iloc[2:4].reset_index(drop=True)
    hold_df = df.iloc[:2].reset_index(drop=True)

    monkeypatch.setattr(
        sample_tht_tripgen, "load_and_split_tht_tripgen", lambda: (train_df, val_df, hold_df)
    )

    transformer = THTTripGenTransformer().fit(train_df)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_tht_tripgen_mappings(run_dir / "mappings_tht_tripgen.json", transformer)

    zone_embeddings = torch.randn(transformer.num_zones, 6)
    cond_card = transformer.conditional_idx_cardinalities

    model = THTTripGenModel(
        num_zones=transformer.num_zones,
        num_time_bins=transformer.num_time_bins,
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
        passenger_cardinality=transformer.passenger_cardinality,
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

    meta = {
        "num_zones": transformer.num_zones,
        "num_time_bins": transformer.num_time_bins,
        "conditional_cardinalities": cond_card,
        "zone_embeddings_source": "Efunc",
        "zone_embeddings_dim": 6,
        "cond_emb_dim": 4,
        "time_emb_dim": 4,
        "origin_emb_dim": 5,
        "context_mlp_hidden": 12,
        "context_mlp_layers": 2,
        "dropout": 0.0,
        "destination_head_type": "embedding_softmax",
        "use_passenger_count": True,
        "passenger_cardinality": transformer.passenger_cardinality,
        "use_total_amount": True,
        "total_amount_sigma_floor": 1e-4,
        "min_r_eps": 1e-6,
        "residual_num_layers": 2,
        "residual_num_bins": 4,
        "residual_context_hidden": 8,
        "residual_min_bin_width": 1e-2,
        "residual_min_bin_height": 1e-2,
        "residual_min_deriv": 1e-2,
        "residual_eps": 1e-6,
    }

    save_checkpoint(
        run_dir / "tht_tripgen.pt",
        model_state=model.state_dict(),
        meta=meta,
        optimizer_state=None,
        epoch=1,
        metrics=None,
    )

    out_path = sample_tht_tripgen.sample_tht_tripgen(
        run_dir=run_dir,
        split="val",
        rows=1,
        temperature=1.0,
        seed=123,
        device=torch.device("cpu"),
        privacy_report=True,
    )

    assert out_path.exists()
    report_path = run_dir / "metrics" / "privacy_report_tht_tripgen_val.json"
    assert report_path.exists()
