import numpy as np
import pandas as pd
import pytest

from data_processing.tht_tripgen_loader import _compute_h_and_r


def test_compute_h_and_r_hierarchical_time():
    datetimes = pd.Series(
        pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 01:30:00",
            "2024-01-01 23:59:59",
        ]
        )
    )
    eps = 1e-6
    h, r = _compute_h_and_r(datetimes, H=24, eps=eps)

    assert h.tolist() == [0, 1, 23]
    assert r[0] == pytest.approx(eps)
    assert r[1] == pytest.approx(0.5, rel=1e-6)

    expected_last = (86399.0 - 23 * 3600.0) / 3600.0
    assert r[2] == pytest.approx(expected_last, rel=1e-6)
    assert np.all(r >= eps)
    assert np.all(r < 1.0 - eps + 1e-12)
