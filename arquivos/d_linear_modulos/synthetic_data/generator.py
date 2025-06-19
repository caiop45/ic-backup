import numpy as np
import pandas as pd
from config import COUNT_NOISE_FRAC, COORD_SIGMA_DEG
from utils.helpers import decode_hour_from_sincos, decode_time_from_sincos
from typing import Sequence

# ---------- amostragem do GMM ---------- #
def synth_samples_cod1(gmm, n_samples, scaler, feature_names):
    synth_scaled = gmm.sample(n_samples).cpu().numpy()
    df = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=feature_names)
    df["hour_of_day"] = decode_time_from_sincos(df["sin_hr"], df["cos_hr"])
    df["trip_count"] = 1
    return df

# ---------- equal-freq por hora ---------- #
def equal_freq(s_df, hour_counts_real, rng):
    parts = []
    for hr, n_real in hour_counts_real.items():
        sub = s_df[s_df["hour_of_day"] == hr]
        if sub.empty:
            continue
        parts.append(
            sub.sample(n=n_real, replace=len(sub) < n_real,
                       random_state=rng.integers(1e6))
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=s_df.columns)

# ---------- perturbação de contagem ---------- #
def perturb_counts(df, rng, frac=COUNT_NOISE_FRAC):
    parts = []
    for hr in range(24):
        sub = df[df["hour_of_day"] == hr]
        if sub.empty:
            continue
        delta  = int(rng.normal(0, len(sub) * frac))
        target = max(1, len(sub) + delta)

        if target < len(sub):
            sub = sub.sample(n=target, replace=False, random_state=rng.integers(1e6))
        elif target > len(sub):
            extra = sub.sample(n=target - len(sub), replace=True,
                               random_state=rng.integers(1e6))
            sub   = pd.concat([sub, extra], ignore_index=True)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=df.columns)

