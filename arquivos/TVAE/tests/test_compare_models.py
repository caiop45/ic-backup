import pandas as pd

from tools.compare_models import DEFAULT_COLUMNS, build_metrics_table


def test_compare_models_table_columns_order():
    baseline_metrics = {
        "eval_split": "hold",
        "n_real": 10,
        "n_synth": 10,
        "w1_tr_te": 0.1,
        "w1_tr_syn": 0.2,
        "w1_te_syn": 0.3,
        "g_tr_te": 70.0,
        "g_tr_syn": 30.0,
        "g_te_syn": 25.0,
        "cov_tr_te": 80.0,
        "cov_tr_syn": 15.0,
        "cov_te_syn": 14.0,
        "dcr_rr_p05": 0.1,
        "dcr_ss_p05": 0.2,
    }
    strategy_metrics = {
        "eval_split": "hold",
        "n_real": 8,
        "n_synth": 8,
        "w1_tr_te": 0.15,
        "w1_tr_syn": 0.25,
        "w1_te_syn": 0.35,
        "g_tr_te": 65.0,
        "g_tr_syn": 28.0,
        "g_te_syn": 24.0,
        "cov_tr_te": 78.0,
        "cov_tr_syn": 12.0,
        "cov_te_syn": 11.0,
        "dcr_rr_p05": 0.11,
        "dcr_ss_p05": 0.21,
    }

    df = build_metrics_table(baseline_metrics, strategy_metrics)

    assert list(df.columns) == DEFAULT_COLUMNS
    assert "dwn_fare_syn_te_r2" in DEFAULT_COLUMNS
    assert "dwn_fare_syn_te_mae" in DEFAULT_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "model"] == "baseline"
    assert df.loc[1, "model"] == "tht_tripgen"
    assert df.loc[0, "w1_tr_syn"] == baseline_metrics["w1_tr_syn"]
    assert df.loc[1, "cov_te_syn"] == strategy_metrics["cov_te_syn"]
