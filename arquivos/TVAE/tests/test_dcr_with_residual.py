import math

import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.metrics import coverage_score, dcr_quantile


def _make_df(r_values: list[float]) -> pd.DataFrame:
    n = len(r_values)
    return pd.DataFrame(
        {
            "dia_da_semana": [0] * n,
            "hora_do_dia": [0] * n,
            "pickup_id": [1] * n,
            "dropoff_id": [2] * n,
            "r": r_values,
        }
    )


def test_dcr_residual_nonzero_when_r_differs() -> None:
    ref_df = _make_df([0.0, 0.2, 0.4, 0.6, 0.8])
    other_df = _make_df([0.1, 0.3, 0.5, 0.7, 0.9])

    dcr = dcr_quantile(
        ref_df,
        other_df,
        alpha=0.05,
        max_samples=None,
        chunk_size=8,
        seed=7,
        w_time=1.0,
        w_space=1.0,
        w_residual=1.0,
        use_residual=True,
        residual_col="r",
    )

    assert dcr > 0.0


def test_dcr_residual_missing_falls_back() -> None:
    ref_df = _make_df([0.0, 0.2, 0.4]).drop(columns=["r"])
    other_df = _make_df([0.1, 0.3, 0.5]).drop(columns=["r"])

    dcr_with = dcr_quantile(
        ref_df,
        other_df,
        alpha=0.05,
        max_samples=None,
        chunk_size=8,
        seed=11,
        w_time=1.0,
        w_space=1.0,
        w_residual=1.0,
        use_residual=True,
        residual_col="r",
    )
    dcr_without = dcr_quantile(
        ref_df,
        other_df,
        alpha=0.05,
        max_samples=None,
        chunk_size=8,
        seed=11,
        w_time=1.0,
        w_space=1.0,
        w_residual=0.0,
        use_residual=False,
        residual_col="r",
    )

    assert dcr_with == dcr_without


def test_coverage_score_with_residual_runs() -> None:
    ref_df = _make_df([0.0, 0.2, 0.4, 0.6])
    other_df = _make_df([0.1, 0.3, 0.5, 0.7])

    cov = coverage_score(
        ref_df,
        other_df,
        k=1,
        max_samples=None,
        seed=3,
        chunk_size=4,
        w_time=1.0,
        w_space=1.0,
        w_residual=1.0,
        use_residual=True,
        residual_col="r",
    )

    assert math.isfinite(float(cov))
    assert 0.0 <= cov <= 100.0
