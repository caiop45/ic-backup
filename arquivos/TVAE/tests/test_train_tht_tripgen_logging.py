import csv

import pandas as pd
import pytest


torch = pytest.importorskip("torch")

import config
import train_tht_tripgen


def test_train_tht_tripgen_logging(tmp_path, monkeypatch):
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
        train_tht_tripgen, "load_and_split_tht_tripgen", lambda: (train_df, val_df, hold_df)
    )

    cache_dir = tmp_path / "topology_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb = torch.randn(4, 6)
    torch.save(emb, cache_dir / "Efunc.pt")

    monkeypatch.setattr(config, "THT_TOPOLOGY_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(config, "SAVE_DATA_DIR", str(tmp_path / "save"))
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "PLOT_DIR", str(tmp_path / "plots"))

    monkeypatch.setattr(config, "THT_EPOCHS", 2)
    monkeypatch.setattr(config, "THT_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "THT_LR", 1e-3)
    monkeypatch.setattr(config, "THT_WEIGHT_DECAY", 0.0)
    monkeypatch.setattr(config, "THT_EVAL_SAMPLE_RATIO", 1.0)
    monkeypatch.setattr(config, "THT_LOG_BATCH_EVERY", 0)
    monkeypatch.setattr(config, "THT_FLOW_DIAG_EVERY_EPOCHS", 0)
    monkeypatch.setattr(config, "THT_MONITOR_METRICS_EVERY_EPOCHS", 0)
    monkeypatch.setattr(config, "LOG_ENABLE_TENSORBOARD", False)
    monkeypatch.setattr(config, "LOG_FLUSH_EVERY", 1)

    train_tht_tripgen.train_tht_tripgen(run_tag="logging", device=torch.device("cpu"))

    output_dir = tmp_path / "save" / "logging"
    csv_path = output_dir / "logs" / "scalars.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    tags = {row["tag"] for row in rows}
    assert "train/nll_total" in tags
    assert "val/nll_total" in tags
