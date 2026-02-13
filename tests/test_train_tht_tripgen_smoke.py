import json

import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from tvae import config
from tvae import train_tht_tripgen


def test_train_tht_tripgen_smoke(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "hour_of_day": [0, 1, 2, 3],
            "r": [0.1, 0.2, 0.3, 0.4],
            "pickup_id": [1, 2, 3, 4],
            "dropoff_id": [2, 3, 4, 1],
            "day_of_week": [0, 1, 2, 3],
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

    monkeypatch.setattr(config, "THT_EPOCHS", 1)
    monkeypatch.setattr(config, "THT_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "THT_LR", 1e-3)
    monkeypatch.setattr(config, "THT_WEIGHT_DECAY", 0.0)
    monkeypatch.setattr(config, "THT_EVAL_SAMPLE_RATIO", 1.0)
    monkeypatch.setattr(config, "COVERAGE_MAX_SAMPLES", 10)
    monkeypatch.setattr(config, "COVERAGE_CHUNK_SIZE", 2)

    train_tht_tripgen.train_tht_tripgen(run_tag="smoke", device=torch.device("cpu"))

    output_dir = tmp_path / "save" / "smoke"
    metrics_path = output_dir / "metrics" / "metrics_tht_tripgen_val.json"
    hold_metrics_path = output_dir / "metrics" / "metrics_tht_tripgen_hold.json"
    checkpoint_path = output_dir / "tht_tripgen.pt"
    mappings_path = output_dir / "mappings_tht_tripgen.json"
    loss_path = output_dir / "loss_tht_tripgen.csv"
    synth_val_path = output_dir / "data" / "synthetic_tht_tripgen_val.csv"
    synth_hold_path = output_dir / "data" / "synthetic_tht_tripgen_hold.csv"

    assert checkpoint_path.exists()
    assert mappings_path.exists()
    assert loss_path.exists()
    assert metrics_path.exists()
    assert hold_metrics_path.exists()
    assert synth_val_path.exists()
    assert synth_hold_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("eval_split") == "val"
    assert "w1_tr_te" in metrics
    assert "g_tr_syn" in metrics
    assert "cov_tr_syn" in metrics
