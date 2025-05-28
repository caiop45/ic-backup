# ──────────────────────────────────────────────────────────────
# Funções utilitárias para balancear o nº de viagens por dia.
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import pandas as pd

def get_min_daily_trips(df: pd.DataFrame) -> int:
    """
    Retorna o menor nº de viagens observado em um único dia.
    Espera colunas: 'tpep_pickup_datetime' e 'num_viagens'.
    """
    if "tpep_pickup_datetime" not in df.columns:
        raise ValueError("'tpep_pickup_datetime' ausente no DataFrame.")
    daily_counts = (
        df.groupby(df["tpep_pickup_datetime"].dt.normalize())["num_viagens"]
          .count()
    )
    return int(daily_counts.min())

def downsample_to_min_daily(
    df: pd.DataFrame,
    min_trips: int,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Garante que **cada** dia contenha exatamente `min_trips` viagens
    (amostragem sem reposição). Mantém todas as colunas originais.
    """
    if min_trips <= 0:
        raise ValueError("`min_trips` deve ser > 0.")
    grouped = df.groupby(df["tpep_pickup_datetime"].dt.normalize(), group_keys=False)
    balanced = grouped.apply(
        lambda x: x.sample(n=min_trips,
                           replace=False,
                           random_state=rng.integers(1_000_000_000))
    )
    return balanced.reset_index(drop=True)
