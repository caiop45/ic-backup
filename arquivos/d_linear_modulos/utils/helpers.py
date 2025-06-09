import numpy as np
import pandas as pd
import config
def decode_hour(sin_s, cos_s):
    """
    Converte seno/cosseno da hora em inteiro de 0-23 h.
    """
    ang = np.mod(np.arctan2(sin_s, cos_s), 2 * np.pi)
    return np.rint(ang * 24 / (2 * np.pi)).astype(int) % 24


def agrupar_viagens_por_local(df: pd.DataFrame) -> pd.DataFrame:
    """
    Soma 'num_viagens' por hora, tendo como colunas finais *os nomes de zona*.
    Se uma zona do CSV não aparecer no df, a coluna é criada com zeros.

    Parâmetros
    ----------
    df : pd.DataFrame
        Deve conter:
        • 'tpep_pickup_datetime'   – tpep_pickup_datetime
        • 'PULocationID'  – **nome da zona** (string)
        • 'num_viagens'   – contagem de viagens

    Retorna
    -------
    pd.DataFrame
        Tabela com 'tpep_pickup_datetime' + uma coluna para cada zona do CSV.
    """
    # ── Lê o CSV de correlação só uma vez ────────────────────────────────
    _cor_df = pd.read_csv(config.ZONE_CRR, dtype={"LocationID": "int32"})
    _EXPECTED_ZONES = _cor_df["zone"].tolist()  # ordem preservada
    _EXPECTED_SET   = set(_EXPECTED_ZONES)      # p/ verificação rápida
    # --------------------------------------------------------------------
    df = df.copy()
    df = df.dropna(subset=["PULocationID"])
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
        values="num_viagens",
        index="tpep_pickup_datetime",
        columns="PULocationID",
        aggfunc="sum",
    )

    # 3. Garante todas as zonas do CSV como colunas
    #    e zera qualquer NaN remanescente
    pivot_df = (
        pivot_df
        .reindex(columns=_EXPECTED_ZONES)  # adiciona zonas ausentes
        .fillna(0)                         # zera horários sem viagens
    )


    # 4. Ajusta formato final
    result_df = pivot_df.reset_index()
    result_df.columns.name = None
    # garante tipo inteiro nas colunas de contagem
    result_df[_EXPECTED_ZONES] = result_df[_EXPECTED_ZONES].astype("int32")

    return result_df

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / denom)