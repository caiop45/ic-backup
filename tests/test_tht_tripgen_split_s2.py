import pandas as pd

from tvae import config
from tvae.data_processing.tht_tripgen_loader import split_tht_tripgen


def test_split_tht_tripgen_s2_no_date_overlap(monkeypatch):
    datetimes = pd.date_range("2024-01-01", periods=10, freq="D").repeat(2)
    df = pd.DataFrame({config.DATETIME_COL: datetimes})

    monkeypatch.setattr(config, "THT_TRAIN_FRAC", 0.5)
    monkeypatch.setattr(config, "THT_VAL_FRAC", 0.3)
    monkeypatch.setattr(config, "THT_SPLIT_SEED", 123)

    train_df, val_df, hold_df = split_tht_tripgen(df, "S2")

    assert len(train_df) + len(val_df) + len(hold_df) == len(df)

    def dates(frame: pd.DataFrame) -> set:
        return set(frame[config.DATETIME_COL].dt.normalize().unique())

    train_dates = dates(train_df)
    val_dates = dates(val_df)
    hold_dates = dates(hold_df)

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(hold_dates)
    assert val_dates.isdisjoint(hold_dates)
