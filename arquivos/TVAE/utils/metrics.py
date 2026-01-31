from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_counts(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / max(total, eps)


def safe_log1p(values: np.ndarray | pd.Series) -> np.ndarray:
    """Safely compute log1p on non-negative values (clip negatives to 0)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    return np.log1p(arr)


def wasserstein_1d_continuous(
    a: np.ndarray, b: np.ndarray, *, grid_size: int = 1001
) -> float:
    """Approximate W1 distance via quantile matching on a fixed grid."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    qs = np.linspace(0.0, 1.0, int(grid_size), dtype=np.float64)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


def ks_statistic_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute KS statistic between two 1D samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    a = np.sort(a)
    b = np.sort(b)
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / float(a.size)
    cdf_b = np.searchsorted(b, values, side="right") / float(b.size)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def quantile_mae(a: np.ndarray, b: np.ndarray, qs: Sequence[float]) -> float:
    """Mean absolute error between quantiles of two samples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 0.0
    q = np.array(list(qs), dtype=np.float64)
    qa = np.quantile(a, q)
    qb = np.quantile(b, q)
    return float(np.mean(np.abs(qa - qb)))


def median_profile_mae(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str,
) -> float:
    """MAE between groupwise medians for two datasets (ignore missing groups)."""
    if real_df.empty or synth_df.empty:
        return 0.0
    real = real_df.groupby(group_col)[value_col].median()
    synth = synth_df.groupby(group_col)[value_col].median()
    idx = real.index.union(synth.index)
    real = real.reindex(idx)
    synth = synth.reindex(idx)
    mask = real.notna() & synth.notna()
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(real[mask] - synth[mask])))


def _align_counts(
    real_counts: pd.Series, synth_counts: pd.Series
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    idx = real_counts.index.union(synth_counts.index)
    real = real_counts.reindex(idx, fill_value=0).to_numpy()
    synth = synth_counts.reindex(idx, fill_value=0).to_numpy()
    labels = [str(x) for x in idx]
    return real, synth, labels


def jsd_counts(real_counts: pd.Series, synth_counts: pd.Series, eps: float = 1e-12) -> float:
    real, synth, _ = _align_counts(real_counts, synth_counts)
    p = _normalize_counts(real)
    q = _normalize_counts(synth)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def chi2_counts(real_counts: pd.Series, synth_counts: pd.Series, eps: float = 1e-12) -> float:
    real, synth, _ = _align_counts(real_counts, synth_counts)
    p = _normalize_counts(real)
    q = _normalize_counts(synth)
    denom = q + eps
    return float(np.sum(((p - q) ** 2) / denom))


def marginal_counts(df: pd.DataFrame, column: str) -> pd.Series:
    return df[column].value_counts().sort_index()


def topk_table(
    real_counts: pd.Series, synth_counts: pd.Series, k: int
) -> pd.DataFrame:
    real_probs = _normalize_counts(real_counts.to_numpy())
    real_probs = pd.Series(real_probs, index=real_counts.index)
    synth_probs = _normalize_counts(synth_counts.to_numpy())
    synth_probs = pd.Series(synth_probs, index=synth_counts.index)

    top_idx = real_probs.sort_values(ascending=False).head(k).index
    real_top = real_probs.reindex(top_idx, fill_value=0)
    synth_top = synth_probs.reindex(top_idx, fill_value=0)

    out = pd.DataFrame(
        {
            "category": top_idx,
            "real_prob": real_top.values,
            "synth_prob": synth_top.values,
            "abs_diff": (real_top - synth_top).abs().values,
        }
    )
    return out


def od_counts(
    df: pd.DataFrame,
    *,
    pickup_col: str = "pickup_id",
    dropoff_col: str = "dropoff_id",
) -> pd.Series:
    """Contagem de pares OD (sem tempo)."""
    if df.empty:
        return pd.Series(dtype="int64")
    key = df[pickup_col].astype(str) + "-" + df[dropoff_col].astype(str)
    return key.value_counts()


def time_counts(
    df: pd.DataFrame,
    *,
    day_col: str = "dia_da_semana",
    hour_col: str = "hora_do_dia",
) -> pd.Series:
    """Contagem de combinações dia × hora (sem espaço)."""
    if df.empty:
        return pd.Series(dtype="int64")
    key = df[day_col].astype(str) + "|" + df[hour_col].astype(str)
    return key.value_counts()


def joint_counts(
    df: pd.DataFrame,
    *,
    od_cols: Tuple[str, str] = ("pickup_id", "dropoff_id"),
    time_cols: Tuple[str, str] = ("dia_da_semana", "hora_do_dia"),
) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="int64")
    tmp = df[list(od_cols) + list(time_cols)].copy()
    tmp["od_key"] = tmp[od_cols[0]].astype(str) + "-" + tmp[od_cols[1]].astype(str)
    tmp["joint_key"] = (
        tmp["od_key"]
        + "|"
        + tmp[time_cols[0]].astype(str)
        + "|"
        + tmp[time_cols[1]].astype(str)
    )
    return tmp["joint_key"].value_counts()


def od_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    pickup_col: str = "pickup_id",
    dropoff_col: str = "dropoff_id",
) -> Dict[str, float]:
    """Métricas apenas para distribuição OD (pickup × dropoff)."""
    real_counts = od_counts(real_df, pickup_col=pickup_col, dropoff_col=dropoff_col)
    synth_counts = od_counts(synth_df, pickup_col=pickup_col, dropoff_col=dropoff_col)

    real_keys = set(real_counts.index)
    synth_keys = set(synth_counts.index)
    inter = real_keys & synth_keys

    coverage_real = len(inter) / max(1, len(real_keys))
    coverage_synth = len(inter) / max(1, len(synth_keys))

    return {
        "od_jsd": jsd_counts(real_counts, synth_counts),
        "od_chi2": chi2_counts(real_counts, synth_counts),
        "od_coverage_real": coverage_real,
        "od_coverage_synth": coverage_synth,
        "od_unique_real": float(len(real_keys)),
        "od_unique_synth": float(len(synth_keys)),
    }


def time_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    day_col: str = "dia_da_semana",
    hour_col: str = "hora_do_dia",
) -> Dict[str, float]:
    """Métricas apenas para distribuição temporal (dia × hora)."""
    real_counts = time_counts(real_df, day_col=day_col, hour_col=hour_col)
    synth_counts = time_counts(synth_df, day_col=day_col, hour_col=hour_col)

    real_keys = set(real_counts.index)
    synth_keys = set(synth_counts.index)
    inter = real_keys & synth_keys

    coverage_real = len(inter) / max(1, len(real_keys))
    coverage_synth = len(inter) / max(1, len(synth_keys))

    return {
        "time_jsd": jsd_counts(real_counts, synth_counts),
        "time_chi2": chi2_counts(real_counts, synth_counts),
        "time_coverage_real": coverage_real,
        "time_coverage_synth": coverage_synth,
        "time_unique_real": float(len(real_keys)),
        "time_unique_synth": float(len(synth_keys)),
    }


def joint_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    od_cols: Tuple[str, str] = ("pickup_id", "dropoff_id"),
    time_cols: Tuple[str, str] = ("dia_da_semana", "hora_do_dia"),
) -> Dict[str, float]:
    """Métricas para distribuição conjunta completa (OD × dia × hora)."""
    real_counts = joint_counts(real_df, od_cols=od_cols, time_cols=time_cols)
    synth_counts = joint_counts(synth_df, od_cols=od_cols, time_cols=time_cols)

    real_keys = set(real_counts.index)
    synth_keys = set(synth_counts.index)
    inter = real_keys & synth_keys

    coverage_real = len(inter) / max(1, len(real_keys))
    coverage_synth = len(inter) / max(1, len(synth_keys))

    # Mode dropping: combinações do real que nunca aparecem no sintético
    missing = real_keys - synth_keys
    mode_dropping_ratio = len(missing) / max(1, len(real_keys))

    # Combinações inválidas: sintético gerou combinações não existentes no real
    invalid = synth_keys - real_keys
    invalid_ratio = len(invalid) / max(1, len(synth_keys))

    metrics = {
        "joint_jsd": jsd_counts(real_counts, synth_counts),
        "joint_chi2": chi2_counts(real_counts, synth_counts),
        "joint_coverage_real": coverage_real,
        "joint_coverage_synth": coverage_synth,
        "joint_unique_real": float(len(real_keys)),
        "joint_unique_synth": float(len(synth_keys)),
        "joint_mode_dropping_ratio": mode_dropping_ratio,
        "joint_invalid_ratio": invalid_ratio,
    }
    return metrics


def compute_fast_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    do_coverage: bool = False,
    coverage_max_samples: int | None = None,
    coverage_k: int | None = None,
    coverage_chunk_size: int | None = None,
    coverage_time_weight: float | None = None,
    coverage_space_weight: float | None = None,
    seed: int | None = None,
) -> Dict[str, float]:
    """Fast metrics without plotting (marginals + OD/time/joint, optional coverage)."""
    import config

    metrics: Dict[str, float] = {}
    metrics["n_real"] = float(len(real_df))
    metrics["n_synth"] = float(len(synth_df))

    for col in config.OUTPUT_COLUMNS:
        real_counts = marginal_counts(real_df, col)
        synth_counts = marginal_counts(synth_df, col)
        metrics[f"{col}_jsd"] = jsd_counts(real_counts, synth_counts)
        metrics[f"{col}_chi2"] = chi2_counts(real_counts, synth_counts)

    metrics.update(od_metrics(real_df, synth_df))
    metrics.update(time_metrics(real_df, synth_df))
    metrics.update(joint_metrics(real_df, synth_df))

    if do_coverage:
        metrics["cov_real_synth"] = coverage_score(
            real_df,
            synth_df,
            k=int(coverage_k if coverage_k is not None else config.COVERAGE_K),
            max_samples=coverage_max_samples
            if coverage_max_samples is not None
            else config.COVERAGE_MAX_SAMPLES,
            seed=int(seed if seed is not None else config.GLOBAL_SEED),
            chunk_size=int(
                coverage_chunk_size
                if coverage_chunk_size is not None
                else config.COVERAGE_CHUNK_SIZE
            ),
            w_time=float(
                coverage_time_weight
                if coverage_time_weight is not None
                else config.COVERAGE_TIME_WEIGHT
            ),
            w_space=float(
                coverage_space_weight
                if coverage_space_weight is not None
                else config.COVERAGE_SPACE_WEIGHT
            ),
        )

    return metrics


def plot_marginal_hist(
    real_counts: pd.Series,
    synth_counts: pd.Series,
    *,
    title: str,
    path: str | Path,
) -> None:
    real, synth, labels = _align_counts(real_counts, synth_counts)
    real = _normalize_counts(real)
    synth = _normalize_counts(synth)

    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, real, width=width, label="real")
    plt.bar(x + width / 2, synth, width=width, label="synth")
    plt.title(title)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_continuous_hist(
    real_vals: np.ndarray,
    synth_vals: np.ndarray,
    *,
    title: str,
    path: str | Path,
    bins: int = 40,
) -> None:
    real_vals = np.asarray(real_vals, dtype=np.float64)
    synth_vals = np.asarray(synth_vals, dtype=np.float64)
    real_vals = real_vals[np.isfinite(real_vals)]
    synth_vals = synth_vals[np.isfinite(synth_vals)]
    if real_vals.size == 0 or synth_vals.size == 0:
        return
    plt.figure(figsize=(10, 4))
    plt.hist(real_vals, bins=bins, density=True, alpha=0.6, label="real")
    plt.hist(synth_vals, bins=bins, density=True, alpha=0.6, label="synth")
    plt.title(title)
    plt.tight_layout()
    plt.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_topk(
    topk_df: pd.DataFrame,
    *,
    title: str,
    path: str | Path,
) -> None:
    labels = [str(x) for x in topk_df["category"].tolist()]
    x = np.arange(len(labels))
    width = 0.4

    plt.figure(figsize=(10, 4))
    plt.bar(x - width / 2, topk_df["real_prob"], width=width, label="real")
    plt.bar(x + width / 2, topk_df["synth_prob"], width=width, label="synth")
    plt.title(title)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.tight_layout()
    plt.legend()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def save_metrics_json(metrics: Dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = pd.Series(metrics).to_json(indent=2)
    path.write_text(text, encoding="utf-8")


def time_key_series(
    df: pd.DataFrame, *, day_col: str = "dia_da_semana", hour_col: str = "hora_do_dia"
) -> np.ndarray:
    day = df[day_col].to_numpy(dtype=np.int64)
    hour = df[hour_col].to_numpy(dtype=np.int64)
    return day * 24 + hour


def wasserstein_1d(
    a: np.ndarray, b: np.ndarray, *, support_size: int | None = None
) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    if support_size is None:
        max_val = int(max(a.max() if a.size else 0, b.max() if b.size else 0))
        support_size = max_val + 1

    counts_a = np.bincount(a, minlength=support_size).astype(np.float64)
    counts_b = np.bincount(b, minlength=support_size).astype(np.float64)

    total_a = counts_a.sum()
    total_b = counts_b.sum()
    if total_a <= 0 or total_b <= 0:
        return 0.0

    p = counts_a / total_a
    q = counts_b / total_b
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def wasserstein_time(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    support_size: int = 7 * 24,
) -> float:
    a = time_key_series(real_df)
    b = time_key_series(synth_df)
    return wasserstein_1d(a, b, support_size=support_size)


def graph_similarity_score(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    pickup_col: str = "pickup_id",
    dropoff_col: str = "dropoff_id",
) -> float:
    if real_df.empty or synth_df.empty:
        return 0.0

    real_counts = (
        real_df.groupby([pickup_col, dropoff_col]).size().astype(np.float64)
    )
    synth_counts = (
        synth_df.groupby([pickup_col, dropoff_col]).size().astype(np.float64)
    )

    idx = real_counts.index.union(synth_counts.index)
    real = real_counts.reindex(idx, fill_value=0).to_numpy()
    synth = synth_counts.reindex(idx, fill_value=0).to_numpy()

    total_real = real.sum()
    total_synth = synth.sum()
    if total_real <= 0 or total_synth <= 0:
        return 0.0

    p = real / total_real
    q = synth / total_synth
    tv = 0.5 * np.sum(np.abs(p - q))
    return float(1.0 - tv)


def _cyc_dist(a: np.ndarray, b: np.ndarray, period: int) -> np.ndarray:
    diff = np.abs(a[:, None] - b[None, :]).astype(np.float32)
    wrap = (period - diff).astype(np.float32)
    return np.minimum(diff, wrap) / (period / 2.0)


def _residual_array(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    values = df[col].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(values, 0.0, 1.0)


def _passenger_array(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values.astype(np.float32)


def _fare_array(df: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in df.columns:
        return None
    return safe_log1p(df[col]).astype(np.float32)


def _pairwise_distance(
    a_day: np.ndarray,
    a_hour: np.ndarray,
    a_pickup: np.ndarray,
    a_dropoff: np.ndarray,
    b_day: np.ndarray,
    b_hour: np.ndarray,
    b_pickup: np.ndarray,
    b_dropoff: np.ndarray,
    *,
    w_time: float,
    w_space: float,
    w_residual: float = 0.0,
    a_r: np.ndarray | None = None,
    b_r: np.ndarray | None = None,
    w_passenger: float = 0.0,
    w_fare: float = 0.0,
    a_pax: np.ndarray | None = None,
    b_pax: np.ndarray | None = None,
    a_fare: np.ndarray | None = None,
    b_fare: np.ndarray | None = None,
    pax_scale: float = 1.0,
    fare_scale: float = 1.0,
    scale_eps: float = 1e-6,
) -> np.ndarray:
    day_dist = _cyc_dist(a_day, b_day, period=7)
    hour_dist = _cyc_dist(a_hour, b_hour, period=24)
    time_dist = day_dist + hour_dist

    pickup_mismatch = (a_pickup[:, None] != b_pickup[None, :]).astype(np.float32)
    dropoff_mismatch = (a_dropoff[:, None] != b_dropoff[None, :]).astype(np.float32)
    space_dist = pickup_mismatch + dropoff_mismatch

    if w_residual > 0.0 and a_r is not None and b_r is not None:
        residual_dist = np.abs(a_r[:, None] - b_r[None, :]).astype(np.float32)
    else:
        residual_dist = 0.0

    if w_passenger > 0.0 and a_pax is not None and b_pax is not None:
        denom = max(float(pax_scale), float(scale_eps))
        passenger_dist = np.abs(a_pax[:, None] - b_pax[None, :]).astype(np.float32) / denom
    else:
        passenger_dist = 0.0

    if w_fare > 0.0 and a_fare is not None and b_fare is not None:
        denom = max(float(fare_scale), float(scale_eps))
        fare_dist = np.abs(a_fare[:, None] - b_fare[None, :]).astype(np.float32) / denom
    else:
        fare_dist = 0.0

    total = (
        w_time * time_dist
        + w_space * space_dist
        + w_residual * residual_dist
        + w_passenger * passenger_dist
        + w_fare * fare_dist
    )
    return total.astype(np.float32)


def distance_matrix(
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    *,
    w_time: float,
    w_space: float,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
    w_passenger: float | None = None,
    w_fare: float | None = None,
    passenger_col: str | None = None,
    fare_col: str | None = None,
    pax_scale: float | None = None,
    fare_scale: float | None = None,
    scale_eps: float | None = None,
) -> np.ndarray:
    """Compute the distance matrix between two datasets.

    Distance definition matches coverage_score:
    - time distance: cyclical day (period=7) + hour (period=24)
    - space distance: pickup mismatch (0/1) + dropoff mismatch (0/1)
    - residual distance: |r_i - r_j| when enabled (r expected in [0,1])
    - total = w_time * time_dist + w_space * space_dist + w_residual * residual_dist
              + w_passenger * passenger_dist + w_fare * fare_dist
    """
    if a_df.empty or b_df.empty:
        return np.zeros((len(a_df), len(b_df)), dtype=np.float32)

    import config

    if use_residual is None:
        use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    if w_residual is None:
        w_residual = float(getattr(config, "COVERAGE_RESIDUAL_WEIGHT", 1.0))
    if w_passenger is None:
        w_passenger = 0.0
    if w_fare is None:
        w_fare = 0.0
    if passenger_col is None:
        passenger_col = str(getattr(config, "PASSENGER_COL", "passenger_count"))
    if fare_col is None:
        fare_col = str(getattr(config, "FARE_COL", "total_amount"))
    if pax_scale is None:
        pax_scale = 1.0
    if fare_scale is None:
        fare_scale = 1.0
    if scale_eps is None:
        scale_eps = float(getattr(config, "DCR_SCALE_EPS", 1e-6))

    a_day = a_df["dia_da_semana"].to_numpy(dtype=np.int64)
    a_hour = a_df["hora_do_dia"].to_numpy(dtype=np.int64)
    a_pickup = a_df["pickup_id"].to_numpy(dtype=np.int64)
    a_dropoff = a_df["dropoff_id"].to_numpy(dtype=np.int64)
    a_r = _residual_array(a_df, residual_col) if use_residual else None
    a_pax = _passenger_array(a_df, passenger_col) if w_passenger > 0.0 else None
    a_fare = _fare_array(a_df, fare_col) if w_fare > 0.0 else None

    b_day = b_df["dia_da_semana"].to_numpy(dtype=np.int64)
    b_hour = b_df["hora_do_dia"].to_numpy(dtype=np.int64)
    b_pickup = b_df["pickup_id"].to_numpy(dtype=np.int64)
    b_dropoff = b_df["dropoff_id"].to_numpy(dtype=np.int64)
    b_r = _residual_array(b_df, residual_col) if use_residual else None
    b_pax = _passenger_array(b_df, passenger_col) if w_passenger > 0.0 else None
    b_fare = _fare_array(b_df, fare_col) if w_fare > 0.0 else None

    if a_r is None or b_r is None or w_residual <= 0.0:
        a_r = None
        b_r = None
        w_residual = 0.0
    if a_pax is None or b_pax is None or w_passenger <= 0.0:
        a_pax = None
        b_pax = None
        w_passenger = 0.0
    if a_fare is None or b_fare is None or w_fare <= 0.0:
        a_fare = None
        b_fare = None
        w_fare = 0.0

    return _pairwise_distance(
        a_day,
        a_hour,
        a_pickup,
        a_dropoff,
        b_day,
        b_hour,
        b_pickup,
        b_dropoff,
        w_time=w_time,
        w_space=w_space,
        w_residual=w_residual,
        a_r=a_r,
        b_r=b_r,
        w_passenger=w_passenger,
        w_fare=w_fare,
        a_pax=a_pax,
        b_pax=b_pax,
        a_fare=a_fare,
        b_fare=b_fare,
        pax_scale=float(pax_scale),
        fare_scale=float(fare_scale),
        scale_eps=float(scale_eps),
    )


def _sample_df(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def coverage_score(
    ref_df: pd.DataFrame,
    other_df: pd.DataFrame,
    *,
    k: int,
    max_samples: int | None,
    seed: int,
    chunk_size: int,
    w_time: float,
    w_space: float,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
    w_passenger: float | None = None,
    w_fare: float | None = None,
    passenger_col: str | None = None,
    fare_col: str | None = None,
    pax_scale: float | None = None,
    fare_scale: float | None = None,
    scale_eps: float | None = None,
) -> float:
    """Coverage score based on nearest-neighbor radii in the reference set.

    Distance includes cyclical day/hour, pickup/dropoff mismatch, and optionally
    residual time r in [0,1], passenger_count, and fare (log1p). Continuous
    terms are normalized by provided scales.
    """
    if ref_df.empty or other_df.empty:
        return 0.0

    import config

    if use_residual is None:
        use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    if w_residual is None:
        w_residual = float(getattr(config, "COVERAGE_RESIDUAL_WEIGHT", 1.0))
    if w_passenger is None:
        w_passenger = 0.0
    if w_fare is None:
        w_fare = 0.0
    if passenger_col is None:
        passenger_col = str(getattr(config, "PASSENGER_COL", "passenger_count"))
    if fare_col is None:
        fare_col = str(getattr(config, "FARE_COL", "total_amount"))
    if pax_scale is None:
        pax_scale = 1.0
    if fare_scale is None:
        fare_scale = 1.0
    if scale_eps is None:
        scale_eps = float(getattr(config, "DCR_SCALE_EPS", 1e-6))

    ref_df = _sample_df(ref_df, max_samples, seed)
    other_df = _sample_df(other_df, max_samples, seed + 1)

    ref_day = ref_df["dia_da_semana"].to_numpy(dtype=np.int64)
    ref_hour = ref_df["hora_do_dia"].to_numpy(dtype=np.int64)
    ref_pickup = ref_df["pickup_id"].to_numpy(dtype=np.int64)
    ref_dropoff = ref_df["dropoff_id"].to_numpy(dtype=np.int64)
    ref_r = _residual_array(ref_df, residual_col) if use_residual else None
    ref_pax = _passenger_array(ref_df, passenger_col) if w_passenger > 0.0 else None
    ref_fare = _fare_array(ref_df, fare_col) if w_fare > 0.0 else None

    other_day = other_df["dia_da_semana"].to_numpy(dtype=np.int64)
    other_hour = other_df["hora_do_dia"].to_numpy(dtype=np.int64)
    other_pickup = other_df["pickup_id"].to_numpy(dtype=np.int64)
    other_dropoff = other_df["dropoff_id"].to_numpy(dtype=np.int64)
    other_r = _residual_array(other_df, residual_col) if use_residual else None
    other_pax = _passenger_array(other_df, passenger_col) if w_passenger > 0.0 else None
    other_fare = _fare_array(other_df, fare_col) if w_fare > 0.0 else None

    if ref_r is None or other_r is None or w_residual <= 0.0:
        ref_r = None
        other_r = None
        w_residual = 0.0
    if ref_pax is None or other_pax is None or w_passenger <= 0.0:
        ref_pax = None
        other_pax = None
        w_passenger = 0.0
    if ref_fare is None or other_fare is None or w_fare <= 0.0:
        ref_fare = None
        other_fare = None
        w_fare = 0.0

    n_ref = len(ref_df)
    if n_ref < 2:
        return 0.0
    k = int(min(k, n_ref - 1))
    if k <= 0:
        return 0.0
    chunk_size = int(max(1, chunk_size))

    radii = np.zeros(n_ref, dtype=np.float32)

    for start in range(0, n_ref, chunk_size):
        end = min(n_ref, start + chunk_size)
        dist = _pairwise_distance(
            ref_day[start:end],
            ref_hour[start:end],
            ref_pickup[start:end],
            ref_dropoff[start:end],
            ref_day,
            ref_hour,
            ref_pickup,
            ref_dropoff,
            w_time=w_time,
            w_space=w_space,
            w_residual=w_residual,
            a_r=None if ref_r is None else ref_r[start:end],
            b_r=ref_r,
            w_passenger=w_passenger,
            w_fare=w_fare,
            a_pax=None if ref_pax is None else ref_pax[start:end],
            b_pax=ref_pax,
            a_fare=None if ref_fare is None else ref_fare[start:end],
            b_fare=ref_fare,
            pax_scale=float(pax_scale),
            fare_scale=float(fare_scale),
            scale_eps=float(scale_eps),
        )
        row_idx = np.arange(start, end)
        dist[np.arange(end - start), row_idx] = np.inf
        kth = np.partition(dist, k - 1, axis=1)[:, k - 1]
        radii[start:end] = kth

    covered = 0
    for start in range(0, n_ref, chunk_size):
        end = min(n_ref, start + chunk_size)
        dist = _pairwise_distance(
            ref_day[start:end],
            ref_hour[start:end],
            ref_pickup[start:end],
            ref_dropoff[start:end],
            other_day,
            other_hour,
            other_pickup,
            other_dropoff,
            w_time=w_time,
            w_space=w_space,
            w_residual=w_residual,
            a_r=None if ref_r is None else ref_r[start:end],
            b_r=other_r,
            w_passenger=w_passenger,
            w_fare=w_fare,
            a_pax=None if ref_pax is None else ref_pax[start:end],
            b_pax=other_pax,
            a_fare=None if ref_fare is None else ref_fare[start:end],
            b_fare=other_fare,
            pax_scale=float(pax_scale),
            fare_scale=float(fare_scale),
            scale_eps=float(scale_eps),
        )
        min_dist = dist.min(axis=1)
        covered += int((min_dist <= radii[start:end]).sum())

    return float(100.0 * covered / max(1, n_ref))


def dcr_quantile(
    ref_df: pd.DataFrame,
    other_df: pd.DataFrame,
    *,
    alpha: float,
    max_samples: int | None,
    chunk_size: int,
    seed: int,
    w_time: float,
    w_space: float,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
    w_passenger: float | None = None,
    w_fare: float | None = None,
    passenger_col: str | None = None,
    fare_col: str | None = None,
    pax_scale: float | None = None,
    fare_scale: float | None = None,
    scale_eps: float | None = None,
) -> float:
    """Compute DCR quantile d_alpha between ref_df and other_df.

    Uses the same distance as coverage_score (time/space + optional residual,
    passenger_count, and fare). Continuous terms are normalized by provided
    scales. Sampling is deterministic given the seed.
    """
    if ref_df.empty or other_df.empty:
        return 0.0

    import config

    if use_residual is None:
        use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    if w_residual is None:
        w_residual = float(getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0))
    if w_passenger is None:
        w_passenger = 0.0
    if w_fare is None:
        w_fare = 0.0
    if passenger_col is None:
        passenger_col = str(getattr(config, "PASSENGER_COL", "passenger_count"))
    if fare_col is None:
        fare_col = str(getattr(config, "FARE_COL", "total_amount"))
    if pax_scale is None:
        pax_scale = 1.0
    if fare_scale is None:
        fare_scale = 1.0
    if scale_eps is None:
        scale_eps = float(getattr(config, "DCR_SCALE_EPS", 1e-6))

    ref_df = _sample_df(ref_df, max_samples, seed)
    other_df = _sample_df(other_df, max_samples, seed + 1)
    if ref_df.empty or other_df.empty:
        return 0.0

    chunk_size = int(max(1, chunk_size))
    min_dists: List[np.ndarray] = []

    ref_day = ref_df["dia_da_semana"].to_numpy(dtype=np.int64)
    ref_hour = ref_df["hora_do_dia"].to_numpy(dtype=np.int64)
    ref_pickup = ref_df["pickup_id"].to_numpy(dtype=np.int64)
    ref_dropoff = ref_df["dropoff_id"].to_numpy(dtype=np.int64)
    ref_r = _residual_array(ref_df, residual_col) if use_residual else None
    ref_pax = _passenger_array(ref_df, passenger_col) if w_passenger > 0.0 else None
    ref_fare = _fare_array(ref_df, fare_col) if w_fare > 0.0 else None

    oth_day = other_df["dia_da_semana"].to_numpy(dtype=np.int64)
    oth_hour = other_df["hora_do_dia"].to_numpy(dtype=np.int64)
    oth_pickup = other_df["pickup_id"].to_numpy(dtype=np.int64)
    oth_dropoff = other_df["dropoff_id"].to_numpy(dtype=np.int64)
    oth_r = _residual_array(other_df, residual_col) if use_residual else None
    oth_pax = _passenger_array(other_df, passenger_col) if w_passenger > 0.0 else None
    oth_fare = _fare_array(other_df, fare_col) if w_fare > 0.0 else None

    if ref_r is None or oth_r is None or w_residual <= 0.0:
        ref_r = None
        oth_r = None
        w_residual = 0.0
    if ref_pax is None or oth_pax is None or w_passenger <= 0.0:
        ref_pax = None
        oth_pax = None
        w_passenger = 0.0
    if ref_fare is None or oth_fare is None or w_fare <= 0.0:
        ref_fare = None
        oth_fare = None
        w_fare = 0.0

    n_ref = len(ref_df)
    n_oth = len(other_df)
    for start in range(0, n_ref, chunk_size):
        end = min(n_ref, start + chunk_size)
        chunk_min = None
        for j in range(0, n_oth, chunk_size):
            j_end = min(n_oth, j + chunk_size)
            dist = _pairwise_distance(
                ref_day[start:end],
                ref_hour[start:end],
                ref_pickup[start:end],
                ref_dropoff[start:end],
                oth_day[j:j_end],
                oth_hour[j:j_end],
                oth_pickup[j:j_end],
                oth_dropoff[j:j_end],
                w_time=w_time,
                w_space=w_space,
                w_residual=w_residual,
                a_r=None if ref_r is None else ref_r[start:end],
                b_r=None if oth_r is None else oth_r[j:j_end],
                w_passenger=w_passenger,
                w_fare=w_fare,
                a_pax=None if ref_pax is None else ref_pax[start:end],
                b_pax=None if oth_pax is None else oth_pax[j:j_end],
                a_fare=None if ref_fare is None else ref_fare[start:end],
                b_fare=None if oth_fare is None else oth_fare[j:j_end],
                pax_scale=float(pax_scale),
                fare_scale=float(fare_scale),
                scale_eps=float(scale_eps),
            )
            min_chunk = dist.min(axis=1)
            if chunk_min is None:
                chunk_min = min_chunk
            else:
                chunk_min = np.minimum(chunk_min, min_chunk)
        if chunk_min is not None:
            min_dists.append(chunk_min)

    if not min_dists:
        return 0.0
    all_min = np.concatenate(min_dists)
    return float(np.quantile(all_min, alpha))


def dcr_within_quantile(
    df: pd.DataFrame,
    *,
    alpha: float,
    max_samples: int | None,
    chunk_size: int,
    seed: int,
    w_time: float,
    w_space: float,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
    w_passenger: float | None = None,
    w_fare: float | None = None,
    passenger_col: str | None = None,
    fare_col: str | None = None,
    pax_scale: float | None = None,
    fare_scale: float | None = None,
    scale_eps: float | None = None,
) -> float:
    """Compute within-set DCR quantile excluding self-matches.

    Uses the same distance as coverage_score (including optional residual,
    passenger_count, and fare). Self-matches are excluded by setting the
    diagonal to +inf for same-chunk comparisons.
    """
    if len(df) < 2:
        return 0.0

    import config

    if use_residual is None:
        use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    if w_residual is None:
        w_residual = float(getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0))
    if w_passenger is None:
        w_passenger = 0.0
    if w_fare is None:
        w_fare = 0.0
    if passenger_col is None:
        passenger_col = str(getattr(config, "PASSENGER_COL", "passenger_count"))
    if fare_col is None:
        fare_col = str(getattr(config, "FARE_COL", "total_amount"))
    if pax_scale is None:
        pax_scale = 1.0
    if fare_scale is None:
        fare_scale = 1.0
    if scale_eps is None:
        scale_eps = float(getattr(config, "DCR_SCALE_EPS", 1e-6))

    df = _sample_df(df, max_samples, seed)
    if len(df) < 2:
        return 0.0

    chunk_size = int(max(1, chunk_size))
    min_dists: List[np.ndarray] = []

    ref_day = df["dia_da_semana"].to_numpy(dtype=np.int64)
    ref_hour = df["hora_do_dia"].to_numpy(dtype=np.int64)
    ref_pickup = df["pickup_id"].to_numpy(dtype=np.int64)
    ref_dropoff = df["dropoff_id"].to_numpy(dtype=np.int64)
    ref_r = _residual_array(df, residual_col) if use_residual else None
    ref_pax = _passenger_array(df, passenger_col) if w_passenger > 0.0 else None
    ref_fare = _fare_array(df, fare_col) if w_fare > 0.0 else None

    if ref_r is None or w_residual <= 0.0:
        ref_r = None
        w_residual = 0.0
    if ref_pax is None or w_passenger <= 0.0:
        ref_pax = None
        w_passenger = 0.0
    if ref_fare is None or w_fare <= 0.0:
        ref_fare = None
        w_fare = 0.0

    n_ref = len(df)
    for start in range(0, n_ref, chunk_size):
        end = min(n_ref, start + chunk_size)
        chunk_min = np.full(end - start, np.inf, dtype=np.float32)
        for j in range(0, n_ref, chunk_size):
            j_end = min(n_ref, j + chunk_size)
            dist = _pairwise_distance(
                ref_day[start:end],
                ref_hour[start:end],
                ref_pickup[start:end],
                ref_dropoff[start:end],
                ref_day[j:j_end],
                ref_hour[j:j_end],
                ref_pickup[j:j_end],
                ref_dropoff[j:j_end],
                w_time=w_time,
                w_space=w_space,
                w_residual=w_residual,
                a_r=None if ref_r is None else ref_r[start:end],
                b_r=None if ref_r is None else ref_r[j:j_end],
                w_passenger=w_passenger,
                w_fare=w_fare,
                a_pax=None if ref_pax is None else ref_pax[start:end],
                b_pax=None if ref_pax is None else ref_pax[j:j_end],
                a_fare=None if ref_fare is None else ref_fare[start:end],
                b_fare=None if ref_fare is None else ref_fare[j:j_end],
                pax_scale=float(pax_scale),
                fare_scale=float(fare_scale),
                scale_eps=float(scale_eps),
            )
            if start == j:
                diag_len = min(end - start, j_end - j)
                idx = np.arange(diag_len)
                dist[idx, idx] = np.inf
            chunk_min = np.minimum(chunk_min, dist.min(axis=1))
        min_dists.append(chunk_min)

    if not min_dists:
        return 0.0
    all_min = np.concatenate(min_dists)
    return float(np.quantile(all_min, alpha))


def rdcr(
    train_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    alpha: float,
    max_samples: int | None,
    chunk_size: int,
    seed: int,
    w_time: float,
    w_space: float,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
    eps: float,
) -> float:
    """Compute rDCR ratio d_alpha(train,synth) / d_alpha(hold,synth)."""
    d_tr = dcr_quantile(
        train_df,
        synth_df,
        alpha=alpha,
        max_samples=max_samples,
        chunk_size=chunk_size,
        seed=seed,
        w_time=w_time,
        w_space=w_space,
        w_residual=w_residual,
        use_residual=use_residual,
        residual_col=residual_col,
    )
    d_hold = dcr_quantile(
        hold_df,
        synth_df,
        alpha=alpha,
        max_samples=max_samples,
        chunk_size=chunk_size,
        seed=seed,
        w_time=w_time,
        w_space=w_space,
        w_residual=w_residual,
        use_residual=use_residual,
        residual_col=residual_col,
    )
    return float(d_tr / max(d_hold, eps))
