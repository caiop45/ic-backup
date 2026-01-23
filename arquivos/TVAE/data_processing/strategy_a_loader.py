"""Strategy A data loader for x~ = (h, r, o, d, u).

Schema:
    h: discrete time bin (0..H-1) stored in "hora_do_dia"
    r: residual within bin in [0,1) stored in "r"
    o: pickup_id
    d: dropoff_id
    u: calendar conditionals (dia_da_semana; optional is_weekend, month)

Splits:
    S1: weekly chronological split (temporal generalization)
    S2: day-level split within the same period (privacy-oriented)
"""
from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np
import pandas as pd

import config


StrategySplit = Literal["S1", "S2"]


def _apply_time_filters_strategy_a(df: pd.DataFrame) -> pd.DataFrame:
    if config.DATETIME_COL not in df.columns:
        raise ValueError(f"Missing datetime column: {config.DATETIME_COL}")

    dt = df[config.DATETIME_COL]
    if config.SA_FILTER_YEAR is not None:
        df = df[dt.dt.year == config.SA_FILTER_YEAR]
        dt = df[config.DATETIME_COL]
    if config.SA_FILTER_MONTHS:
        df = df[dt.dt.month.isin(config.SA_FILTER_MONTHS)]
        dt = df[config.DATETIME_COL]
    if config.SA_FILTER_DOW_MIN is not None and config.SA_FILTER_DOW_MAX is not None:
        df = df[dt.dt.dayofweek.between(config.SA_FILTER_DOW_MIN, config.SA_FILTER_DOW_MAX)]
        dt = df[config.DATETIME_COL]
    if config.SA_FILTER_START_DATE is not None:
        df = df[dt >= pd.to_datetime(config.SA_FILTER_START_DATE)]
        dt = df[config.DATETIME_COL]
    if config.SA_FILTER_END_DATE is not None:
        df = df[dt <= pd.to_datetime(config.SA_FILTER_END_DATE)]

    return df


def _seconds_since_midnight(datetime_series: pd.Series) -> np.ndarray:
    dt = datetime_series.dt
    seconds = (
        dt.hour.astype("int64") * 3600
        + dt.minute.astype("int64") * 60
        + dt.second.astype("int64")
    ).astype("float64")
    seconds += dt.microsecond.astype("float64") / 1e6
    return seconds


def _compute_h_and_r(
    datetime_series: pd.Series, H: int, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    if H <= 0:
        raise ValueError("H must be a positive integer")
    if eps <= 0 or eps >= 0.5:
        raise ValueError("eps must be in (0, 0.5)")

    s = _seconds_since_midnight(datetime_series)
    bin_width = 86400.0 / float(H)
    h = np.floor(s / bin_width).astype("int64")
    h = np.clip(h, 0, H - 1)
    r = (s - h * bin_width) / bin_width
    r = np.clip(r, eps, 1.0 - eps).astype("float32")
    return h.astype("int64"), r


def _strategy_a_columns() -> List[str]:
    cols = list(config.SA_STRATEGY_A_COLUMNS)
    required = ["hora_do_dia", "r", "pickup_id", "dropoff_id", "dia_da_semana"]
    missing = [col for col in required if col not in cols]
    if missing:
        raise ValueError(f"SA_STRATEGY_A_COLUMNS missing required columns: {missing}")

    if config.SA_USE_WEEKEND and "is_weekend" not in cols:
        cols.append("is_weekend")
    if config.SA_USE_MONTH and "month" not in cols:
        cols.append("month")

    return cols


def load_raw_data_strategy_a() -> pd.DataFrame:
    df = pd.read_parquet(config.REAL_DATA_PATH)
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL], errors="coerce")
    df = df.dropna(
        subset=[config.DATETIME_COL, config.PICKUP_ID_COL, config.DROPOFF_ID_COL]
    )

    df = _apply_time_filters_strategy_a(df)
    df = df.copy()

    h, r = _compute_h_and_r(
        df[config.DATETIME_COL],
        H=config.SA_TIME_BINS_H,
        eps=config.SA_MIN_R_EPS,
    )

    df["hora_do_dia"] = h.astype("int64")
    df["r"] = r.astype("float32")
    df["dia_da_semana"] = df[config.DATETIME_COL].dt.dayofweek.astype("int64")
    df["pickup_id"] = df[config.PICKUP_ID_COL].astype("int64")
    df["dropoff_id"] = df[config.DROPOFF_ID_COL].astype("int64")

    if config.SA_USE_WEEKEND:
        df["is_weekend"] = (df["dia_da_semana"] >= 5).astype("int64")
    if config.SA_USE_MONTH:
        df["month"] = df[config.DATETIME_COL].dt.month.astype("int64")

    keep_cols = [config.DATETIME_COL] + _strategy_a_columns()
    df = df[keep_cols]
    df = df.dropna(subset=_strategy_a_columns())

    return df


def _split_weekly_chronological(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    datetime_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values(datetime_col).reset_index(drop=True)
    iso_week = df_sorted[datetime_col].dt.isocalendar()
    df_sorted["_year_week"] = (
        iso_week["year"].astype(str) + "-" + iso_week["week"].astype(str).str.zfill(2)
    )

    week_sizes = df_sorted.groupby("_year_week").size()
    total_rows = len(df_sorted)
    if total_rows == 0:
        empty = df_sorted.drop(columns=["_year_week"], errors="ignore")
        return empty.copy(), empty.copy(), empty.copy()

    cum_rows = week_sizes.cumsum()
    train_weeks = cum_rows[cum_rows / total_rows <= train_frac].index.tolist()

    remaining = cum_rows.loc[~cum_rows.index.isin(train_weeks)]
    val_weeks = remaining[remaining / total_rows <= train_frac + val_frac].index.tolist()

    hold_weeks = [w for w in week_sizes.index if w not in train_weeks + val_weeks]

    df_train = (
        df_sorted[df_sorted["_year_week"].isin(train_weeks)]
        .drop(columns=["_year_week"])
        .reset_index(drop=True)
    )
    df_val = (
        df_sorted[df_sorted["_year_week"].isin(val_weeks)]
        .drop(columns=["_year_week"])
        .reset_index(drop=True)
    )
    df_hold = (
        df_sorted[df_sorted["_year_week"].isin(hold_weeks)]
        .drop(columns=["_year_week"])
        .reset_index(drop=True)
    )

    return df_train, df_val, df_hold


def _split_by_day(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    datetime_col: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_work = df.copy()
    df_work["_date_key"] = df_work[datetime_col].dt.normalize()

    unique_days = df_work["_date_key"].drop_duplicates().sort_values().to_numpy()
    num_days = len(unique_days)
    if num_days == 0:
        empty = df_work.drop(columns=["_date_key"], errors="ignore")
        return empty.copy(), empty.copy(), empty.copy()

    rng = np.random.default_rng(seed)
    shuffled_days = rng.permutation(unique_days)

    train_days_count = int(np.floor(train_frac * num_days))
    val_days_count = int(np.floor(val_frac * num_days))

    train_days = set(shuffled_days[:train_days_count])
    val_days = set(
        shuffled_days[train_days_count : train_days_count + val_days_count]
    )
    hold_days = set(shuffled_days[train_days_count + val_days_count :])

    df_train = (
        df_work[df_work["_date_key"].isin(train_days)]
        .drop(columns=["_date_key"])
        .reset_index(drop=True)
    )
    df_val = (
        df_work[df_work["_date_key"].isin(val_days)]
        .drop(columns=["_date_key"])
        .reset_index(drop=True)
    )
    df_hold = (
        df_work[df_work["_date_key"].isin(hold_days)]
        .drop(columns=["_date_key"])
        .reset_index(drop=True)
    )

    return df_train, df_val, df_hold


def split_strategy_a(
    df: pd.DataFrame, strategy: StrategySplit
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.DATETIME_COL not in df.columns:
        raise ValueError(f"Missing datetime column: {config.DATETIME_COL}")

    if strategy not in ("S1", "S2"):
        raise ValueError("strategy must be 'S1' or 'S2'")

    if strategy == "S1":
        return _split_weekly_chronological(
            df,
            train_frac=config.SA_TRAIN_FRAC,
            val_frac=config.SA_VAL_FRAC,
            datetime_col=config.DATETIME_COL,
        )

    return _split_by_day(
        df,
        train_frac=config.SA_TRAIN_FRAC,
        val_frac=config.SA_VAL_FRAC,
        datetime_col=config.DATETIME_COL,
        seed=config.SA_SPLIT_SEED,
    )


def load_and_split_strategy_a() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data_strategy_a()
    train_df, val_df, hold_df = split_strategy_a(df, config.SA_SPLIT_STRATEGY)

    cols = _strategy_a_columns()
    drop_cols = [config.DATETIME_COL]

    train_df = train_df.drop(columns=drop_cols, errors="ignore")[cols].reset_index(
        drop=True
    )
    val_df = val_df.drop(columns=drop_cols, errors="ignore")[cols].reset_index(
        drop=True
    )
    hold_df = hold_df.drop(columns=drop_cols, errors="ignore")[cols].reset_index(
        drop=True
    )

    return train_df, val_df, hold_df
