import numpy as np
import pandas as pd
from config import COUNT_NOISE_FRAC, COORD_SIGMA_DEG
from utils.helpers import decode_hour

# ---------- amostragem do GMM ---------- #
def synth_samples_cod1(gmm, n_samples, scaler, feature_names):
    synth_scaled = gmm.sample(n_samples).cpu().numpy()
    df = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=feature_names)
    df["hora_do_dia"] = decode_hour(df["sin_hr"], df["cos_hr"])
    df["num_viagens"] = 1
    return df

# ---------- equal-freq por hora ---------- #
def equal_freq(s_df, hour_counts_real, rng):
    parts = []
    for hr, n_real in hour_counts_real.items():
        sub = s_df[s_df["hora_do_dia"] == hr]
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
        sub = df[df["hora_do_dia"] == hr]
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

# ---------- jitter de coordenadas ---------- #
def jitter_coords(df, rng, sigma=COORD_SIGMA_DEG):
    df_jit = df.copy()
    for c in ["PU_latitude", "PU_longitude", "DO_latitude", "DO_longitude"]:
        if c in df_jit.columns:
            df_jit[c] += rng.normal(0, sigma, size=len(df_jit))
    return df_jit

# ---------- quantile-mapping opcional ---------- #
def qmap(col_s, col_r):
    if col_s.empty:
        return pd.Series(dtype=col_s.dtype)
    if col_r.empty:
        return pd.Series(np.nan, index=col_s.index, dtype=np.float64)

    idx  = col_s.rank(method="first").astype(int) - 1
    real = np.sort(col_r.values)
    scaled = np.floor(idx.values * len(real) / len(col_s)).astype(int)
    return pd.Series(real[np.clip(scaled, 0, len(real)-1)], index=col_s.index)

def apply_qmap(s_df, r_df):
    coord_cols = ["PU_longitude", "PU_latitude", "DO_longitude", "DO_latitude"]
    s = s_df.copy()
    if r_df.empty:
        for c in coord_cols: s[c] = np.nan
        return s
    for c in coord_cols:
        s[c] = qmap(s[c], r_df[c].dropna()) if c in r_df.columns else np.nan
    return s
