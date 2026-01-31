import math
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from utils.evaluation import compute_metrics


def _make_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    real_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1, 2, 3, 4, 0, 1, 2],
            "hora_do_dia": [0, 1, 2, 3, 4, 5, 6, 7],
            "pickup_id": [1, 2, 3, 1, 2, 3, 1, 2],
            "dropoff_id": [2, 3, 1, 2, 3, 1, 2, 3],
        }
    )
    synth_df = pd.DataFrame(
        {
            "dia_da_semana": [0, 1, 2, 3, 4],
            "hora_do_dia": [1, 2, 3, 4, 5],
            "pickup_id": [1, 2, 3, 1, 2],
            "dropoff_id": [2, 3, 1, 2, 3],
        }
    )
    return real_df, synth_df


def test_rr_baselines_keys_and_ranges(tmp_path: Path) -> None:
    real_df, synth_df = _make_eval_frames()
    output_dir = tmp_path / "out"
    plot_dir = tmp_path / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(
        real_df,
        synth_df,
        order_key="order_1",
        output_dir=output_dir,
        plot_dir=plot_dir,
        compute_rr_baselines=True,
        random_state=123,
    )

    expected_keys = {
        "od_jsd_rr",
        "od_chi2_rr",
        "od_coverage_real_rr",
        "od_coverage_synth_rr",
        "joint_jsd_rr",
        "joint_chi2_rr",
        "joint_coverage_real_rr",
        "joint_coverage_synth_rr",
        "joint_invalid_ratio_rr",
        "joint_mode_dropping_ratio_rr",
    }

    for key in expected_keys:
        assert key in metrics
        assert math.isfinite(float(metrics[key]))

    for key in ("od_coverage_real_rr", "od_coverage_synth_rr"):
        val = float(metrics[key])
        assert 0.0 <= val <= 1.0

    for key in ("joint_coverage_real_rr", "joint_coverage_synth_rr"):
        val = float(metrics[key])
        assert 0.0 <= val <= 1.0


def test_rr_baselines_deterministic_seed(tmp_path: Path) -> None:
    real_df, synth_df = _make_eval_frames()

    output_dir_1 = tmp_path / "out1"
    plot_dir_1 = tmp_path / "plots1"
    output_dir_2 = tmp_path / "out2"
    plot_dir_2 = tmp_path / "plots2"
    for path in (output_dir_1, plot_dir_1, output_dir_2, plot_dir_2):
        path.mkdir(parents=True, exist_ok=True)

    metrics_1 = compute_metrics(
        real_df,
        synth_df,
        order_key="order_1",
        output_dir=output_dir_1,
        plot_dir=plot_dir_1,
        compute_rr_baselines=True,
        random_state=999,
    )
    metrics_2 = compute_metrics(
        real_df,
        synth_df,
        order_key="order_1",
        output_dir=output_dir_2,
        plot_dir=plot_dir_2,
        compute_rr_baselines=True,
        random_state=999,
    )

    rr_keys = [key for key in metrics_1.keys() if key.endswith("_rr")]
    assert rr_keys, "Expected RR metrics to be present."
    for key in rr_keys:
        assert metrics_1[key] == metrics_2[key]
