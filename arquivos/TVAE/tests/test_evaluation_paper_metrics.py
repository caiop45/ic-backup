import math

import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.evaluation import compute_paper_metrics


def test_compute_paper_metrics_keys_and_types():
    train_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1, 2],
            "hora_do_dia": [0, 1, 2],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [2, 3, 1],
        }
    )
    eval_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1, 2],
            "hora_do_dia": [1, 2, 3],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 1, 2],
        }
    )
    synth_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1, 2],
            "hora_do_dia": [2, 3, 4],
            "pickup_id": [1, 3, 2],
            "dropoff_id": [2, 1, 3],
        }
    )

    metrics = compute_paper_metrics(train_df, eval_df, synth_df)

    expected_keys = {
        "w1_tr_te",
        "w1_tr_syn",
        "w1_te_syn",
        "g_tr_te",
        "g_tr_syn",
        "g_te_syn",
        "cov_tr_te",
        "cov_tr_syn",
        "cov_te_syn",
        "coverage_k",
        "coverage_max_samples",
        "coverage_time_weight",
        "coverage_space_weight",
    }

    for key in expected_keys:
        assert key in metrics
        assert math.isfinite(float(metrics[key]))
