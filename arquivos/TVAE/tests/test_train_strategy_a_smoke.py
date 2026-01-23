import json

import pandas as pd
import pytest


torch = pytest.importorskip("torch")

import config
import train_strategy_a


def test_train_strategy_a_smoke(tmp_path, monkeypatch):
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
        train_strategy_a, "load_and_split_strategy_a", lambda: (train_df, val_df, hold_df)
    )

    cache_dir = tmp_path / "topology_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb = torch.randn(4, 6)
    torch.save(emb, cache_dir / "Efunc.pt")

    monkeypatch.setattr(config, "SA_TOPOLOGY_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(config, "SAVE_DATA_DIR", str(tmp_path / "save"))
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(config, "PLOT_DIR", str(tmp_path / "plots"))

    monkeypatch.setattr(config, "SA_EPOCHS", 1)
    monkeypatch.setattr(config, "SA_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "SA_LR", 1e-3)
    monkeypatch.setattr(config, "SA_WEIGHT_DECAY", 0.0)
    monkeypatch.setattr(config, "SA_EVAL_SAMPLE_RATIO", 1.0)
    monkeypatch.setattr(config, "COVERAGE_MAX_SAMPLES", 10)
    monkeypatch.setattr(config, "COVERAGE_CHUNK_SIZE", 2)

    train_strategy_a.train_strategy_a(run_tag="smoke", device=torch.device("cpu"))

    output_dir = tmp_path / "save" / "smoke"
    metrics_path = output_dir / "metrics_strategy_a.json"
    hold_metrics_path = output_dir / "metrics_strategy_a_hold.json"
    checkpoint_path = output_dir / "strategy_a.pt"
    mappings_path = output_dir / "mappings_strategy_a.json"
    loss_path = output_dir / "loss_strategy_a.csv"
    synth_path = output_dir / "synthetic_strategy_a_hold.csv"

    assert checkpoint_path.exists()
    assert mappings_path.exists()
    assert loss_path.exists()
    assert metrics_path.exists()
    assert hold_metrics_path.exists()
    assert synth_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "w1_tr_te" in metrics
    assert "g_tr_syn" in metrics
    assert "cov_tr_syn" in metrics
