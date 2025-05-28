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
    dados_reais_dlinear_input = df[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()

    return df, dados_reais_gmm, dados_reais_dlinear_input, hour_counts_dict_real, GMM_FEATURES
