import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.metrics import dcr_quantile, rdcr


def test_dcr_quantile_simple():
    ref_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 0],
            "hora_do_dia": [0, 1],
            "pickup_id": [1, 1],
            "dropoff_id": [2, 2],
        }
    )
    other_df = pd.DataFrame(
        {
            "dia_da_semana": [0],
            "hora_do_dia": [0],
            "pickup_id": [1],
            "dropoff_id": [2],
        }
    )

    dcr = dcr_quantile(
        ref_df,
        other_df,
        alpha=1.0,
        max_samples=None,
        chunk_size=2,
        seed=0,
        w_time=1.0,
        w_space=1.0,
    )

    assert dcr == pytest.approx(1.0 / 12.0)


def test_rdcr_ratio():
    train_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 0],
            "hora_do_dia": [0, 1],
            "pickup_id": [1, 1],
            "dropoff_id": [2, 2],
        }
    )
    hold_df = pd.DataFrame(
        {
            "dia_da_semana": [0],
            "hora_do_dia": [5],
            "pickup_id": [1],
            "dropoff_id": [2],
        }
    )
    synth_df = pd.DataFrame(
        {
            "dia_da_semana": [0],
            "hora_do_dia": [0],
            "pickup_id": [1],
            "dropoff_id": [2],
        }
    )

    ratio = rdcr(
        train_df,
        hold_df,
        synth_df,
        alpha=1.0,
        max_samples=None,
        chunk_size=2,
        seed=0,
        w_time=1.0,
        w_space=1.0,
        eps=1e-12,
    )

    assert ratio == pytest.approx(0.2)
