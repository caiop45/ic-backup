import math

import numpy as np
import pandas as pd
import pytest


def _modules():
    pytest.importorskip("torch")
    from tvae import config
    from tvae.utils.evaluation import compute_paper_metrics
    from tvae.utils.metrics import dcr_quantile

    return config, compute_paper_metrics, dcr_quantile


def _base_df(n: int, fare: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour_of_day": [10] * n,
            "day_of_week": [2] * n,
            "pickup_id": [1] * n,
            "dropoff_id": [3] * n,
            "r": np.linspace(0.1, 0.2, n),
            "passenger_count": [2] * n,
            "total_amount": [fare] * n,
        }
    )


def test_extended_dcr_nonzero_when_fare_differs():
    _, _, dcr_quantile = _modules()
    train_df = _base_df(5, fare=10.0)
    synth_df = _base_df(5, fare=12.0)

    base = dcr_quantile(
        train_df,
        synth_df,
        alpha=0.05,
        max_samples=None,
        chunk_size=8,
        seed=0,
        w_time=0.0,
        w_space=0.0,
        w_residual=0.0,
    )
    ext = dcr_quantile(
        train_df,
        synth_df,
        alpha=0.05,
        max_samples=None,
        chunk_size=8,
        seed=0,
        w_time=0.0,
        w_space=0.0,
        w_residual=0.0,
        w_fare=1.0,
        fare_col="total_amount",
        fare_scale=1.0,
    )
    assert base == 0.0
    assert ext > 0.0


def test_extended_dcr_scale_fallback(monkeypatch):
    config, compute_paper_metrics, _ = _modules()
    train_df = _base_df(6, fare=10.0)
    hold_df = _base_df(6, fare=10.0)
    synth_df = _base_df(6, fare=10.0)

    monkeypatch.setattr(config, "EVAL_ENABLE_EXTENDED_PRIVACY", True)
    monkeypatch.setattr(config, "DCR_ALPHAS", [0.05])
    monkeypatch.setattr(config, "DCR_W_TIME", 0.0)
    monkeypatch.setattr(config, "DCR_W_SPACE", 0.0)
    monkeypatch.setattr(config, "DCR_W_R", 0.0)
    monkeypatch.setattr(config, "DCR_W_PAX", 0.0)
    monkeypatch.setattr(config, "DCR_W_FARE", 1.0)
    monkeypatch.setattr(config, "DCR_FARE_SCALE_METHOD", "iqr")
    monkeypatch.setattr(config, "DCR_SCALE_EPS", 1e-6)

    metrics = compute_paper_metrics(train_df, hold_df, synth_df)
    assert "dcr_tr_syn_p05_ext" in metrics
    assert math.isfinite(float(metrics["dcr_tr_syn_p05_ext"]))
