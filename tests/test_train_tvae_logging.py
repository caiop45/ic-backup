import csv

import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from tvae import config
from tvae import train_tvae
from tvae.data_processing.transformer import CategoricalTransformer


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_of_day": [0, 1, 2, 0, 1, 2],
            "day_of_week": [0, 1, 2, 0, 1, 2],
            "pickup_id": [1, 2, 3, 1, 2, 3],
            "dropoff_id": [3, 1, 2, 3, 1, 2],
        }
    )


def test_train_tvae_logging(tmp_path, monkeypatch):
    df = _tiny_df()
    train_df = df.iloc[:4].reset_index(drop=True)
    val_df = df.iloc[4:6].reset_index(drop=True)
    hold_df = df.iloc[:0].reset_index(drop=True)

    monkeypatch.setattr(config, "EPOCHS", 2)
    monkeypatch.setattr(config, "PATIENCE", 10)
    monkeypatch.setattr(config, "BATCH_SIZE", 2)
    monkeypatch.setattr(config, "ENCODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "DECODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "LATENT_DIM", 4)
    monkeypatch.setattr(config, "PAIR_KL_WEIGHT", 0.0)
    monkeypatch.setattr(config, "PICKUP_KL_WEIGHT", 0.0)
    monkeypatch.setattr(config, "TVAE_MONITOR_METRICS_EVERY_EPOCHS", 0)
    monkeypatch.setattr(config, "LOG_ENABLE_TENSORBOARD", False)
    monkeypatch.setattr(config, "LOG_FLUSH_EVERY", 1)

    transformer = CategoricalTransformer(config.OUTPUT_COLUMNS).fit(train_df)

    output_dir = tmp_path / "run"
    log_dir = tmp_path / "logs"
    plot_dir = tmp_path / "plots"

    order_key = config.FIXED_ORDER_KEY
    order = config.ORDERS[order_key]

    train_tvae.train_single_order(
        order_key,
        order,
        device=torch.device("cpu"),
        train_df=train_df,
        val_df=val_df,
        hold_df=hold_df,
        transformer=transformer,
        output_dir=output_dir,
        log_dir=log_dir,
        plot_dir=plot_dir,
    )

    csv_path = output_dir / "logs" / "scalars.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    tags = {row["tag"] for row in rows}
    assert "train/loss" in tags
    assert "val/loss" in tags
