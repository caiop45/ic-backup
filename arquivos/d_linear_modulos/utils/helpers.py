import numpy as np
import pandas as pd
import config
import random
import torch

def quarter_hour_index(dt_series: pd.Series) -> pd.Series:
    """Return the quarter-hour index (0-95) for a datetime series."""
    dt = pd.to_datetime(dt_series)
    return dt.dt.hour * 4 + dt.dt.minute // 15

#Retorna apenas hora
def decode_hour_from_sincos(sin_values, cos_values):
    """Convert hour sin/cos values to integers from 0 to 23."""
    ang = np.mod(np.arctan2(sin_values, cos_values), 2 * np.pi)
    return np.rint(ang * 24 / (2 * np.pi)).astype(int) % 24

#Retorna hora:minuto(15min intervalo)

def decode_time_from_sincos(sin_values, cos_values):
    """Convert hour sin/cos values to pandas timestamps rounded to 15 minutes."""
    # 1. Ângulo ∈ [0, 2π)
    ang = np.mod(np.arctan2(sin_values, cos_values), 2 * np.pi)

    # 2. Minutos decimais no dia (0-1440)
    total_minutes = ang * (24 * 60) / (2 * np.pi)

    # 3. Piso para o múltiplo de 15 min mais próximo
    floored_minutes = (total_minutes // 15) * 15         # já arredonda para baixo
    floored_minutes = floored_minutes.astype(int)        # converte p/ int64

    # 4. Converte p/ datetime64[ns]
    #    origin='unix' => 1970-01-01 00:00; unit='m' => deslocamento em minutos
    return pd.to_datetime(floored_minutes, unit="m", origin="unix")
  
def group_trips_by_zone(
    df: pd.DataFrame,
    expected_zones: list[str] | None = None,
) -> pd.DataFrame:
    """
     Soma 'trip_count' por intervalo de 15 minutos, tendo como colunas finais
    *os nomes de zona*.

    Parâmetros
    ----------
    df : pd.DataFrame
        Deve conter:
        • 'tpep_pickup_datetime'   – tpep_pickup_datetime
        • 'PULocationID'  – **nome da zona** (string)
        • 'trip_count'   – contagem de viagens

    Retorna
    -------
    pd.DataFrame
        Tabela com 'tpep_pickup_datetime' + uma coluna para cada zona do CSV.
    """
    # ── Lê o CSV de correlação só uma vez ────────────────────────────────
    _cor_df = pd.read_csv(
        config.ZONE_CORRELATION_CSV, dtype={"LocationID": "int32"}
    )
    _ALL_ZONES = _cor_df["zone"].tolist()  # ordem preservada
    if expected_zones is None:
        expected_zones = _ALL_ZONES
    _EXPECTED_SET = set(_ALL_ZONES)  # p/ verificação rápida
    # --------------------------------------------------------------------
    df = df.copy()
    df = df.dropna(subset=["PULocationID"])
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.floor("15min")
    # 1. Verifica se todas as zonas do df estão mapeadas no CSV
    zonas_df = set(df["PULocationID"].unique())
    zonas_desconhecidas = zonas_df - _EXPECTED_SET
    if zonas_desconhecidas:
        raise ValueError(
            f"As seguintes zonas não constam no CSV de correlação: "
            f"{sorted(zonas_desconhecidas)}"
        )
    # 2. Pivot table: soma de viagens por hora × zona
    pivot_df = pd.pivot_table(
        df,
        values="trip_count",
        index="tpep_pickup_datetime",
        columns="PULocationID",
        aggfunc="sum",
    )
    full_range = pd.date_range(
        start=pivot_df.index.min(),
        end=pivot_df.index.max(),
        freq="15min",
        name=pivot_df.index.name,
    )
    pivot_df = pivot_df.reindex(full_range, fill_value=0)

    # 3. Garante todas as zonas do CSV como colunas
    #    e zera qualquer NaN remanescente
    pivot_df = (
        pivot_df
        .reindex(columns=expected_zones)  # adiciona zonas ausentes
        .fillna(0)                         # zera horários sem viagens
    )


    # 4. Ajusta formato final
    result_df = pivot_df.reset_index()
    result_df.columns.name = None
    # garante tipo inteiro nas colunas de contagem
    result_df[expected_zones] = result_df[expected_zones].astype("int32")

    return result_df

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)


def set_global_seed(seed: int) -> None:
    """Propaga a mesma seed para todos os geradores de números aleatórios."""
    random.seed(seed)                 # módulo random do Python
    np.random.seed(seed)              # NumPy
    torch.manual_seed(seed)           # PyTorch (CPU)
    torch.cuda.manual_seed_all(seed)  # PyTorch (GPU, se houver)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False