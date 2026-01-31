import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

try:  # pragma: no cover - exercised when sklearn is available
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

from utils.downstream_eval import FareEvalInputs, compute_downstream_fare_metrics


def _make_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, size=n)
    dow = rng.integers(0, 6, size=n)
    pickup = rng.integers(1, 5, size=n)
    dropoff = rng.integers(1, 5, size=n)
    r = rng.random(n)
    passenger = rng.integers(1, 4, size=n)
    total_amount = (
        2.5 * hour
        + 1.2 * dow
        + 0.8 * pickup
        + 0.5 * dropoff
        + 3.0 * r
        + 1.5 * passenger
    )
    return pd.DataFrame(
        {
            "hora_do_dia": hour,
            "dia_da_semana": dow,
            "pickup_id": pickup,
            "dropoff_id": dropoff,
            "r": r,
            "passenger_count": passenger,
            "total_amount": total_amount,
        }
    )


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_downstream_fare_eval_metrics_present():
    train_df = _make_df(200, seed=1)
    eval_df = _make_df(120, seed=2)
    synth_df = _make_df(150, seed=3)

    inputs = FareEvalInputs(
        train_df=train_df, eval_df=eval_df, synth_df=synth_df, eval_split="hold"
    )
    metrics = compute_downstream_fare_metrics(
        inputs, train_rows=100, test_rows=80, model_name="gbr", seed=123
    )

    assert metrics["skipped_fare"] is False
    for key in (
        "dwn_fare_tr_tr",
        "dwn_fare_tr_te",
        "dwn_fare_tr_syn",
        "dwn_fare_syn_tr",
        "dwn_fare_syn_te",
        "dwn_fare_syn_syn",
    ):
        assert key in metrics
        payload = metrics[key]
        assert math.isfinite(float(payload["r2"]))
        assert math.isfinite(float(payload["mae"]))
        assert math.isfinite(float(payload["rmse"]))


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_downstream_fare_eval_missing_label_skips():
    train_df = _make_df(50, seed=1).drop(columns=["total_amount"])
    eval_df = _make_df(50, seed=2)
    synth_df = _make_df(50, seed=3)

    inputs = FareEvalInputs(
        train_df=train_df, eval_df=eval_df, synth_df=synth_df, eval_split="val"
    )
    metrics = compute_downstream_fare_metrics(inputs, seed=123)

    assert metrics["skipped_fare"] is True
    assert "missing_columns" in str(metrics.get("skip_reason", ""))
