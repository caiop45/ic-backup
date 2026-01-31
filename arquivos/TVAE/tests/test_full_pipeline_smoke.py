import sys
from pathlib import Path

import pandas as pd
import pytest


torch = pytest.importorskip("torch")

import config
import recalc_downstream_tht_tripgen
import recalc_metrics_hold
import recalc_metrics_tht_tripgen_hold
import train_tht_tripgen
import train_tvae
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from tools import run_full_pipeline


def _tiny_tvae_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2, 3],
            "dia_da_semana": [0, 1, 2, 3],
            "pickup_id": [1, 2, 3, 4],
            "dropoff_id": [2, 3, 4, 1],
        }
    )


def _tiny_strategy_df() -> pd.DataFrame:
    return pd.DataFrame(
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


def test_full_pipeline_smoke(tmp_path, monkeypatch):
    tvae_df = _tiny_tvae_df()
    train_df = tvae_df.iloc[:3].reset_index(drop=True)
    val_df = tvae_df.iloc[2:4].reset_index(drop=True)
    hold_df = tvae_df.iloc[:2].reset_index(drop=True)

    monkeypatch.setattr(train_tvae, "load_and_split", lambda: (train_df, val_df, hold_df))
    monkeypatch.setattr(recalc_metrics_hold, "load_and_split", lambda: (train_df, val_df, hold_df))

    sa_df = _tiny_strategy_df()
    sa_train = sa_df.iloc[:3].reset_index(drop=True)
    sa_val = sa_df.iloc[2:4].reset_index(drop=True)
    sa_hold = sa_df.iloc[:2].reset_index(drop=True)

    monkeypatch.setattr(
        train_tht_tripgen, "load_and_split_tht_tripgen", lambda: (sa_train, sa_val, sa_hold)
    )
    monkeypatch.setattr(
        recalc_metrics_tht_tripgen_hold,
        "load_and_split_tht_tripgen",
        lambda: (sa_train, sa_val, sa_hold),
    )
    monkeypatch.setattr(
        recalc_downstream_tht_tripgen,
        "load_and_split_tht_tripgen",
        lambda: (sa_train, sa_val, sa_hold),
    )

    cache_dir = tmp_path / "topology_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    transformer = THTTripGenTransformer().fit(sa_train)
    emb = torch.randn(transformer.num_zones, 6)
    torch.save(emb, cache_dir / "Efunc.pt")

    monkeypatch.setattr(config, "THT_TOPOLOGY_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(config, "SAVE_DATA_DIR", str(tmp_path / "save"))
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "PLOT_DIR", str(tmp_path / "plots"))

    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "BATCH_SIZE", 2)
    monkeypatch.setattr(config, "ENCODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "DECODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "LATENT_DIM", 4)
    monkeypatch.setattr(config, "PAIR_KL_WEIGHT", 0.0)
    monkeypatch.setattr(config, "PICKUP_KL_WEIGHT", 0.0)

    monkeypatch.setattr(config, "THT_EPOCHS", 1)
    monkeypatch.setattr(config, "THT_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "THT_LR", 1e-3)
    monkeypatch.setattr(config, "THT_WEIGHT_DECAY", 0.0)

    baseline_dir = tmp_path / "save" / "baseline_smoke"
    strategy_dir = tmp_path / "save" / "strategy_smoke"
    comparison_dir = tmp_path / "comparison"

    argv = [
        "run_full_pipeline.py",
        "--baseline-run-dir",
        str(baseline_dir),
        "--tht-tripgen-run-dir",
        str(strategy_dir),
        "--comparison-dir",
        str(comparison_dir),
        "--force-train",
        "--downstream-train-rows",
        "50",
        "--downstream-test-rows",
        "50",
        "--downstream-model",
        "linear",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_full_pipeline.main()

    table_path = comparison_dir / "metrics_table.csv"
    assert table_path.exists()
    table = pd.read_csv(table_path)
    assert set(table["model"].tolist()) == {"baseline", "tht_tripgen"}

    downstream_path = strategy_dir / "metrics" / "downstream_tht_tripgen_hold.json"
    assert downstream_path.exists()
