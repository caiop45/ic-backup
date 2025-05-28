import numpy as np
import pandas as pd

def make_date_sampler(df_real, ts_col="tpep_pickup_datetime", seed=None):
    """
    Retorna função sample_date(hora) que preserva a distribuição
    de datas reais para cada hora.
    """
    df = df_real.copy()
    df["hora"] = df[ts_col].dt.hour
    df["data"] = df[ts_col].dt.normalize()
    rng = np.random.default_rng(seed or 12345)

    prob = {
        h: (sub["data"].value_counts().sort_index().index.to_numpy(),
            sub["data"].value_counts(normalize=True).sort_index().values)
        for h, sub in df.groupby("hora")
    }

    def sample_date(h):
        if h in prob and len(prob[h][0]):
            dates, probs = prob[h]                     # desempacota
            return rng.choice(dates, p=probs)         # usa p=probs
        available = [hh for hh in prob if len(prob[hh][0])]
        if available:
            fallback = rng.choice(available)
            dates, probs = prob[fallback]
            return rng.choice(dates, p=probs)
        return pd.NaT
    return sample_date
