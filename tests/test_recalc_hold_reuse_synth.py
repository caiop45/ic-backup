import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from tvae import config
from tvae import recalc_metrics_hold
from tvae import train_tvae
from tvae.data_processing.transformer import CategoricalTransformer
from tvae.utils.serialization import save_mappings


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_of_day": [0, 1, 2],
            "day_of_week": [0, 1, 2],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [2, 3, 1],
        }
    )


def test_recalc_hold_reuses_synth(tmp_path, monkeypatch):
    train_df = _tiny_df()
    hold_df = _tiny_df()

    monkeypatch.setattr(recalc_metrics_hold, "load_and_split", lambda: (train_df, train_df, hold_df))

    transformer = CategoricalTransformer(config.OUTPUT_COLUMNS).fit(train_df)
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_mappings(run_dir / "mappings_order_1.json", transformer)

    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    synth_path = data_dir / "synthetic_tvae_order_1_hold.csv"
    synth_df = hold_df.copy()
    synth_df.to_csv(synth_path, index=False)
    mtime_before = synth_path.stat().st_mtime

    def _fail_sample(*_args, **_kwargs):
        raise AssertionError("sampling should not be called")

    monkeypatch.setattr(train_tvae, "_sample_synthetic", _fail_sample)

    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "recalc_metrics_hold.py",
        "--run-dir",
        str(run_dir),
        "--order-key",
        "order_1",
        "--no-plots",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    recalc_metrics_hold.main()

    assert synth_path.stat().st_mtime == mtime_before

    metrics_path = out_dir / "metrics_tvae_order_1_hold.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics.get("eval_split") == "hold"
