import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from models.strategy_a import StrategyAModel
from utils.serialization import save_checkpoint, save_strategy_a_mappings

import config
import sample_strategy_a
from data_processing.strategy_a_transformer import StrategyATransformer


def test_sample_strategy_a_outputs_csv(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2, 3],
            "r": [0.1, 0.2, 0.3, 0.4],
            "pickup_id": [1, 2, 3, 4],
            "dropoff_id": [2, 3, 4, 1],
            "dia_da_semana": [0, 1, 2, 3],
        }
    )
    train_df = df.iloc[:3].reset_index(drop=True)
    val_df = df.iloc[2:4].reset_index(drop=True)
    hold_df = df.iloc[:2].reset_index(drop=True)

    monkeypatch.setattr(
        sample_strategy_a, "load_and_split_strategy_a", lambda: (train_df, val_df, hold_df)
    )

    transformer = StrategyATransformer().fit(train_df)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_strategy_a_mappings(run_dir / "mappings_strategy_a.json", transformer)

    zone_embeddings = torch.randn(transformer.num_zones, 6)
    cond_card = transformer.conditional_idx_cardinalities

    model = StrategyAModel(
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
        run_dir / "strategy_a.pt",
        model_state=model.state_dict(),
        meta=meta,
        optimizer_state=None,
        epoch=1,
        metrics=None,
    )

    out_path = sample_strategy_a.sample_strategy_a(
        run_dir=run_dir,
        split="val",
        rows=1,
        temperature=1.0,
        seed=123,
        device=torch.device("cpu"),
    )

    assert out_path.exists()
    out_df = pd.read_csv(out_path)
    assert list(out_df.columns) == config.OUTPUT_COLUMNS
    assert len(out_df) == 1
