import math

import numpy as np
import pandas as pd
import pytest


def _metrics():
    pytest.importorskip("torch")
    from utils.metrics import (
        ks_statistic_1d,
        median_profile_mae,
        quantile_mae,
        safe_log1p,
        wasserstein_1d_continuous,
    )

    return ks_statistic_1d, median_profile_mae, quantile_mae, safe_log1p, wasserstein_1d_continuous


def test_continuous_metrics_identical():
    ks_statistic_1d, _, quantile_mae, _, wasserstein_1d_continuous = _metrics()
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])

    assert math.isclose(wasserstein_1d_continuous(a, b), 0.0, abs_tol=1e-9)
    assert math.isclose(ks_statistic_1d(a, b), 0.0, abs_tol=1e-9)
    assert math.isclose(quantile_mae(a, b, [0.1, 0.5, 0.9]), 0.0, abs_tol=1e-9)


def test_continuous_metrics_shifted():
    ks_statistic_1d, _, quantile_mae, _, wasserstein_1d_continuous = _metrics()
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = a + 1.0
    assert wasserstein_1d_continuous(a, b) > 0.0
    assert ks_statistic_1d(a, b) > 0.0
    assert quantile_mae(a, b, [0.1, 0.5, 0.9]) > 0.0


def test_safe_log1p_clips_negatives():
    _, _, _, safe_log1p, _ = _metrics()
    values = np.array([-5.0, 0.0, 3.0])
    out = safe_log1p(values)
    assert out[0] == 0.0
    assert out[1] == 0.0
    assert out[2] > 0.0


def test_median_profile_mae():
    _, median_profile_mae, _, _, _ = _metrics()
    real = pd.DataFrame({"hora_do_dia": [0, 0, 1], "val": [10.0, 12.0, 20.0]})
    synth = pd.DataFrame({"hora_do_dia": [0, 1], "val": [11.0, 18.0]})
    mae = median_profile_mae(real, synth, group_col="hora_do_dia", value_col="val")
    assert mae > 0.0
