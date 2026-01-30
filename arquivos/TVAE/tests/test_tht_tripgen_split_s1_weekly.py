import pandas as pd

import config
from data_processing.tht_tripgen_loader import split_tht_tripgen


def test_split_tht_tripgen_s1_weekly_chronological(monkeypatch):
    datetimes = pd.date_range("2024-01-01", periods=21, freq="D")
    df = pd.DataFrame({config.DATETIME_COL: datetimes})

    monkeypatch.setattr(config, "THT_TRAIN_FRAC", 0.4)
    monkeypatch.setattr(config, "THT_VAL_FRAC", 0.3)

    train_df, val_df, hold_df = split_tht_tripgen(df, "S1")

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(hold_df) > 0
    assert len(train_df) + len(val_df) + len(hold_df) == len(df)

    train_max = train_df[config.DATETIME_COL].max()
    val_min = val_df[config.DATETIME_COL].min()
    val_max = val_df[config.DATETIME_COL].max()
    hold_min = hold_df[config.DATETIME_COL].min()

    assert train_max < val_min
    assert val_max < hold_min
