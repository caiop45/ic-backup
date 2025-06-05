import numpy as np
import pandas as pd
from config import COUNT_NOISE_FRAC, COORD_SIGMA_DEG
from utils.helpers import decode_hour
from typing import Sequence

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

import numpy as np
import pandas as pd
from typing import Sequence

def adjust_counts_by_group(
    s_df: pd.DataFrame,
    r_df: pd.DataFrame,
    rng: np.random.Generator,
    group_cols: Sequence[str] = ("PULocationID", "DOLocationID", "hora_do_dia"),
    tol: float = 0.10,
) -> pd.DataFrame:
    """
    Equal-freq + perturb_counts:
    Ajusta o dataframe sintético de modo que, para cada tripla
    (PULocationID, DOLocationID, hora_do_dia), o total de num_viagens
    não ultrapasse (1 + tol) · n_real(g).

    • Se n_real(g) == 0                ➜ descarta todo o grupo sintético  
    • Se n_synth(g) > (1+tol)·n_real(g)➜ remove linhas aleatórias até o limite  
    • Caso contrário                   ➜ mantém o grupo sem alteração
    """

    # ---------- verificações iniciais ----------
    if s_df.empty:
        print("[WARN] DataFrame sintético está vazio. Nada a ajustar.")
        return s_df.copy()

    # ---------- normaliza tipos ----------
    for col in ("PULocationID", "DOLocationID"):
        s_df[col] = s_df[col].astype(str)
        r_df[col] = r_df[col].astype(str)

    keep_cols = list(group_cols) + ["num_viagens"]
    s_df = s_df.loc[:, s_df.columns.isin(keep_cols)].copy()
    r_df = r_df.loc[:, r_df.columns.isin(keep_cols)].copy()

    # ---------- total de viagens por tripla nos dados reais ----------
    real_counts = (
        r_df.groupby(list(group_cols))["num_viagens"]
            .sum()
            .astype(int)
    )
    print(f"[INFO] Grupos reais computados: {len(real_counts):,}")

    # ---------- percorre cada tripla nos dados sintéticos ----------
    parts = []
    for g, sub in s_df.groupby(list(group_cols), group_keys=False):
        n_real  = real_counts.get(g, 0)
        n_synth = sub["num_viagens"].sum()

        # Grupo inexistente nos dados reais ➜ descarta
        if n_real == 0:
            continue

        allowed_max = int(np.ceil(n_real * (1 + tol)))

        # Se exceder o limite ➜ remove linhas aleatórias
        if n_synth > allowed_max:
            frac = allowed_max / n_synth  # fração de linhas a manter
            sub  = sub.sample(
                frac=frac,
                replace=False,
                random_state=rng.integers(1e6),
            )

        parts.append(sub)

    # ---------- concatena resultado ----------
    s_adj = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=s_df.columns)

    # ---------- estatísticas finais ----------
    total_synth = s_adj["num_viagens"].sum()
    total_real  = r_df["num_viagens"].sum()
    diff_pct    = 0 if total_real == 0 else (total_synth / total_real - 1)

    print(
        f"\n[COMPARAÇÃO] num_viagens total\n"
        f"  • Sintético ajustado: {total_synth:,}\n"
        f"  • Real             : {total_real:,}\n"
        f"  • Diferença        : {diff_pct:+.2%}"
    )

    return s_adj


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
