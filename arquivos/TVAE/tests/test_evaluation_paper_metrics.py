import math

import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.evaluation import compute_paper_metrics


def test_compute_paper_metrics_keys_and_types():
    train_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [0, 1, 2],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [2, 3, 1],
        }
    )
    eval_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [1, 2, 3],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 1, 2],
        }
    )
    synth_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [2, 3, 4],
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
        "dcr_tr_syn_p05",
        "dcr_hold_syn_p05",
        "rdcr_p05",
        "coverage_k",
        "coverage_max_samples",
        "coverage_time_weight",
        "coverage_space_weight",
    }

    for key in expected_keys:
        assert key in metrics
        assert math.isfinite(float(metrics[key]))


def test_compute_paper_metrics_within_dcr_keys():
    train_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [0, 1, 2],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [2, 3, 1],
        }
    )
    eval_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [1, 2, 3],
            "pickup_id": [1, 2, 3],
            "dropoff_id": [3, 1, 2],
        }
    )
    synth_df = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "hour_of_day": [2, 3, 4],
            "pickup_id": [1, 3, 2],
            "dropoff_id": [2, 1, 3],
        }
    )

    metrics = compute_paper_metrics(train_df, eval_df, synth_df, include_within=True)

    assert "dcr_rr_p05" in metrics
    assert "dcr_ss_p05" in metrics
    assert math.isfinite(float(metrics["dcr_rr_p05"]))
    assert math.isfinite(float(metrics["dcr_ss_p05"]))
