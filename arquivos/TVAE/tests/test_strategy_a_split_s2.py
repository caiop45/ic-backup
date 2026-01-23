import pandas as pd

import config
from data_processing.strategy_a_loader import split_strategy_a


def test_split_strategy_a_s2_no_date_overlap(monkeypatch):
    datetimes = pd.date_range("2024-01-01", periods=10, freq="D").repeat(2)
    df = pd.DataFrame({config.DATETIME_COL: datetimes})

    monkeypatch.setattr(config, "SA_TRAIN_FRAC", 0.5)
    monkeypatch.setattr(config, "SA_VAL_FRAC", 0.3)
    monkeypatch.setattr(config, "SA_SPLIT_SEED", 123)

    train_df, val_df, hold_df = split_strategy_a(df, "S2")

    assert len(train_df) + len(val_df) + len(hold_df) == len(df)

    def dates(frame: pd.DataFrame) -> set:
        return set(frame[config.DATETIME_COL].dt.normalize().unique())

    train_dates = dates(train_df)
    val_dates = dates(val_df)
    hold_dates = dates(hold_df)

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(hold_dates)
    assert val_dates.isdisjoint(hold_dates)
