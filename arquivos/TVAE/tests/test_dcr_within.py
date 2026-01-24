import pytest
import pandas as pd

pytest.importorskip("torch")

from utils.metrics import dcr_within_quantile


def test_dcr_within_quantile_simple():
    df = pd.DataFrame(
        {
            "dia_da_semana": [0, 0],
            "hora_do_dia": [0, 1],
            "pickup_id": [1, 1],
            "dropoff_id": [2, 2],
        }
    )

    dcr = dcr_within_quantile(
        df,
        alpha=1.0,
        max_samples=None,
        chunk_size=2,
        seed=0,
        w_time=1.0,
        w_space=1.0,
    )

    assert dcr == pytest.approx(1.0 / 12.0)
