from __future__ import annotations

from typing import Tuple

import pandas as pd

from config import (
    DATETIME_COL,
    DROPOFF_ID_COL,
    FILTER_DOW_MAX,
    FILTER_DOW_MIN,
    FILTER_END_DATE,
    FILTER_MONTHS,
    FILTER_START_DATE,
    FILTER_YEAR,
    GLOBAL_SEED,
    PICKUP_ID_COL,
    REAL_DATA_PATH,
    TRAIN_FRAC,
    VAL_FRAC,
)


def _apply_time_filters(df: pd.DataFrame) -> pd.DataFrame:
    if DATETIME_COL not in df.columns:
        raise ValueError(f"Missing datetime column: {DATETIME_COL}")

    dt = df[DATETIME_COL]
    if FILTER_YEAR is not None:
        df = df[dt.dt.year == FILTER_YEAR]
        dt = df[DATETIME_COL]
    if FILTER_MONTHS:
        df = df[dt.dt.month.isin(FILTER_MONTHS)]
        dt = df[DATETIME_COL]
    if FILTER_DOW_MIN is not None and FILTER_DOW_MAX is not None:
        df = df[dt.dt.dayofweek.between(FILTER_DOW_MIN, FILTER_DOW_MAX)]
        dt = df[DATETIME_COL]
    if FILTER_START_DATE is not None:
        df = df[dt >= pd.to_datetime(FILTER_START_DATE)]
        dt = df[DATETIME_COL]
    if FILTER_END_DATE is not None:
        df = df[dt <= pd.to_datetime(FILTER_END_DATE)]

    return df


def load_raw_data() -> pd.DataFrame:
    df = pd.read_parquet(REAL_DATA_PATH)
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")
    df = df.dropna(subset=[DATETIME_COL, PICKUP_ID_COL, DROPOFF_ID_COL])

    df = _apply_time_filters(df)
    df = df.copy()
    df["hora_do_dia"] = df[DATETIME_COL].dt.hour.astype("int64")
    df["dia_da_semana"] = df[DATETIME_COL].dt.dayofweek.astype("int64")
    df["pickup_id"] = df[PICKUP_ID_COL].astype("int64")
    df["dropoff_id"] = df[DROPOFF_ID_COL].astype("int64")

    keep_cols = [DATETIME_COL, "hora_do_dia", "dia_da_semana", "pickup_id", "dropoff_id"]
    df = df[keep_cols]
    df = df.dropna(subset=["hora_do_dia", "dia_da_semana", "pickup_id", "dropoff_id"])

    return df


def split_dataset_weekly(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    datetime_col: str = DATETIME_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(datetime_col).reset_index(drop=True)

    iso_week = df[datetime_col].dt.isocalendar()
    df["_year_week"] = (
        iso_week["year"].astype(str) + "-" + iso_week["week"].astype(str).str.zfill(2)
    )

    week_sizes = df.groupby("_year_week").size()
    total_rows = len(df)

    cum_rows = week_sizes.cumsum()
    train_weeks = cum_rows[cum_rows / total_rows <= train_frac].index.tolist()

    cum_rows_after_train = cum_rows.loc[~cum_rows.index.isin(train_weeks)]
    val_weeks = cum_rows_after_train[
        cum_rows_after_train / total_rows <= train_frac + val_frac
    ].index.tolist()

    hold_weeks = [w for w in week_sizes.index if w not in train_weeks + val_weeks]

    df_train = (
        df[df["_year_week"].isin(train_weeks)]
        .drop(columns="_year_week")
        .reset_index(drop=True)
    )
    df_val = (
        df[df["_year_week"].isin(val_weeks)]
        .drop(columns="_year_week")
        .reset_index(drop=True)
    )
    df_hold = (
        df[df["_year_week"].isin(hold_weeks)]
        .drop(columns="_year_week")
        .reset_index(drop=True)
    )

    return df_train, df_val, df_hold


def split_dataset_random(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = GLOBAL_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    # Hold vazio: tudo que sobra vai para o val quando TRAIN_FRAC + VAL_FRAC >= 1
    if train_frac + val_frac >= 1.0:
        df_val = df.iloc[n_train:].reset_index(drop=True)
        df_hold = df.iloc[0:0].reset_index(drop=True)
    else:
        df_hold = df.iloc[n_train + n_val :].reset_index(drop=True)
    return df_train, df_val, df_hold


def load_and_split() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data()
    # train_df, val_df, hold_df = split_dataset_weekly(df)
    train_df, val_df, hold_df = split_dataset_random(df)
    train_df = train_df.drop(columns=[DATETIME_COL])
    val_df = val_df.drop(columns=[DATETIME_COL])
    hold_df = hold_df.drop(columns=[DATETIME_COL])
    return train_df, val_df, hold_df
