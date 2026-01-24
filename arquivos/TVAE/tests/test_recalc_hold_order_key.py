import json
import sys

import pandas as pd
import pytest

pytest.importorskip("torch")

import config
import recalc_metrics_hold
import train_tvae
from data_processing.transformer import CategoricalTransformer
from utils.serialization import save_mappings


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hora_do_dia": [0, 1, 2],
            "dia_da_semana": [0, 1, 2],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [2, 3, 1],
        }
    )


def test_recalc_hold_order_key_paths(tmp_path, monkeypatch):
    train_df = _tiny_df()
    hold_df = _tiny_df()

    monkeypatch.setattr(recalc_metrics_hold, "load_and_split", lambda: (train_df, train_df, hold_df))

    transformer = CategoricalTransformer(config.OUTPUT_COLUMNS).fit(train_df)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_mappings(run_dir / "mappings_order_2.json", transformer)

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    synth_path = data_dir / "synthetic_tvae_order_2_hold.csv"
    hold_df.to_csv(synth_path, index=False)

    ckpt_path = run_dir / "tvae_order_2.pt"
    ckpt_path.write_bytes(b"stub")

    def _fail_sample(*_args, **_kwargs):
        raise AssertionError("sampling should not be called")

    monkeypatch.setattr(train_tvae, "_sample_synthetic", _fail_sample)

    argv = [
        "recalc_metrics_hold.py",
        "--run-dir",
        str(run_dir),
        "--order-key",
        "order_2",
        "--no-plots",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    recalc_metrics_hold.main()

    metrics_path = run_dir / "metrics" / "metrics_tvae_order_2_hold.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("eval_split") == "hold"
