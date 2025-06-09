import numpy as np
import pandas as pd
from config import REAL_DATA_PATH

def load_real_data():
    """
    Carrega o parquet, filtra período/semana e devolve:
    - dados_reais_orig               (DataFrame completo filtrado)
    - dados_reais_gmm                (features para GMM)
    - dados_reais_dlinear_input      (3 colunas p/ DLinear)
    - hour_counts_dict_real          ({hora: nº de viagens})
    - GMM_FEATURES                   (lista de colunas para o GMM)
    """
    df = pd.read_parquet(REAL_DATA_PATH)
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # 2024-01/02 | seg-qua
    df = df[
        (df["tpep_pickup_datetime"].dt.year == 2024) &
        (df["tpep_pickup_datetime"].dt.month.isin([1, 2])) &
        (df["tpep_pickup_datetime"].dt.dayofweek.between(0, 2))
    ]

    df["hora_do_dia"] = df["tpep_pickup_datetime"].dt.hour
    df["num_viagens"] = 1

    hour_counts = df.groupby("hora_do_dia")["num_viagens"].sum().sort_index()
    hour_counts_dict_real = hour_counts.to_dict()

    df["sin_hr"] = np.sin(2 * np.pi * df["hora_do_dia"] / 24)
    df["cos_hr"] = np.cos(2 * np.pi * df["hora_do_dia"] / 24)

    GMM_FEATURES = [
        "sin_hr", "cos_hr",
        "PU_longitude", "PU_latitude",
        "DO_longitude", "DO_latitude",
    ]
    dados_reais_gmm = df[GMM_FEATURES].dropna().astype(np.float32)
  # dados_reais_dlinear_input = df[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()
    dados_reais_dlinear_input = df.copy()
    return df, dados_reais_gmm, dados_reais_dlinear_input, hour_counts_dict_real, GMM_FEATURES

def split_dataset_weekly(
    df: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
    datetime_col: str = "tpep_pickup_datetime",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide um DataFrame temporal em (treino, validação, hold-out) sem
    quebrar semanas. A divisão é feita no 1º dia útil **anterior** ao
    ponto-alvo, garantindo proporções próximas às desejadas.

    Parâmetros
    ----------
    df : DataFrame já ordenado ou não – será ordenado internamente.
    train_frac : fração aproximada para treino.
    val_frac   : fração aproximada para validação.
    datetime_col : nome da coluna com datas (dtype datetime64).

    Retorna
    -------
    (df_train, df_val, df_hold)
    """
    # 1) Ordena cronologicamente
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 2) Marca a semana ISO (YYYY-WW)
    iso_week = df[datetime_col].dt.isocalendar()
    df["_year_week"] = iso_week["year"].astype(str) + "-" + iso_week["week"].astype(str).str.zfill(2)

    # 3) Calcula tamanho cumulativo por semana
    week_sizes = df.groupby("_year_week").size()
    total_rows = len(df)

    cum_rows = week_sizes.cumsum()
    train_weeks = cum_rows[cum_rows / total_rows <= train_frac].index.tolist()

    cum_rows_after_train = cum_rows.loc[~cum_rows.index.isin(train_weeks)]
    val_weeks = cum_rows_after_train[cum_rows_after_train / total_rows <= train_frac + val_frac].index.tolist()

    hold_weeks = [w for w in week_sizes.index if w not in train_weeks + val_weeks]

    # 4) Constrói os datasets finais
    df_train = df[df["_year_week"].isin(train_weeks)].drop(columns="_year_week").reset_index(drop=True)
    df_val   = df[df["_year_week"].isin(val_weeks)].drop(columns="_year_week").reset_index(drop=True)
    df_hold  = df[df["_year_week"].isin(hold_weeks)].drop(columns="_year_week").reset_index(drop=True)

    return df_train, df_val, df_hold
