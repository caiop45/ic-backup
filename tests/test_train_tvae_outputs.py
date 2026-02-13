import csv
import json

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


@pytest.mark.parametrize("hold_rows", [0, 2])
def test_train_tvae_outputs(tmp_path, monkeypatch, hold_rows):
    df = _tiny_df()
    train_df = df.iloc[:4].reset_index(drop=True)
    val_df = df.iloc[4:6].reset_index(drop=True)
    hold_df = df.iloc[:hold_rows].reset_index(drop=True)

    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "BATCH_SIZE", 2)
    monkeypatch.setattr(config, "PATIENCE", 1)
    monkeypatch.setattr(config, "ENCODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "DECODER_HIDDEN_DIMS", (16,))
    monkeypatch.setattr(config, "LATENT_DIM", 4)
    monkeypatch.setattr(config, "EVAL_SAMPLE_RATIO", 1.0)
    monkeypatch.setattr(config, "MAX_EVAL_SAMPLES", None)
    monkeypatch.setattr(config, "COVERAGE_MAX_SAMPLES", 10)
    monkeypatch.setattr(config, "COVERAGE_CHUNK_SIZE", 2)
    monkeypatch.setattr(config, "PAIR_KL_WEIGHT", 0.0)
    monkeypatch.setattr(config, "PICKUP_KL_WEIGHT", 0.0)

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

    model_key = f"tvae_{order_key}"
    metrics_dir = output_dir / "metrics"
    data_dir = output_dir / "data"

    val_metrics_path = metrics_dir / f"metrics_{model_key}_val.json"
    val_synth_path = data_dir / f"synthetic_{model_key}_val.csv"

    assert val_metrics_path.exists()
    assert val_synth_path.exists()

    val_metrics = json.loads(val_metrics_path.read_text(encoding="utf-8"))
    assert val_metrics.get("eval_split") == "val"

    hold_metrics_path = metrics_dir / f"metrics_{model_key}_hold.json"
    hold_synth_path = data_dir / f"synthetic_{model_key}_hold.csv"

    if hold_rows > 0:
        assert hold_metrics_path.exists()
        assert hold_synth_path.exists()
        hold_metrics = json.loads(hold_metrics_path.read_text(encoding="utf-8"))
        assert hold_metrics.get("eval_split") == "hold"
    else:
        assert not hold_metrics_path.exists()
        assert not hold_synth_path.exists()

    scalars_path = output_dir / "logs" / "scalars.csv"
    assert scalars_path.exists()
    with scalars_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    tags = {row["tag"] for row in rows}
    assert "train/ce_pickup_id" in tags or "val/acc_dropoff_id" in tags
