from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Iterable, List, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _normalize_counts(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / max(total, eps)

def dummify(
    X: np.ndarray, *, cat_indexes: List[int], divide_by: float = 0.0, drop_first: bool = False
) -> Tuple[np.ndarray, List[str], List[str]]:
    df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    df_names_before = list(df.columns)
    for i in cat_indexes:
        df = pd.get_dummies(
            df,
            columns=[str(i)],
            prefix=str(i),
            dtype="float",
            drop_first=drop_first,
        )
        if divide_by > 0:
            filter_col = [col for col in df if col.startswith(str(i) + "_")]
            df[filter_col] = df[filter_col] / divide_by
    df_names_after = list(df.columns)
    return df.to_numpy(), df_names_before, df_names_after


def minmax_scale_dummy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    cat_indexes: List[int] | None = None,
    mask: np.ndarray | None = None,
    *,
    divide_by: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, object, List[str] | None, List[str] | None]:
    if cat_indexes is None:
        cat_indexes = []

    # Avoid mutating caller data
    X_train_ = np.array(X_train, dtype=np.float64, copy=True)
    X_test_ = np.array(X_test, dtype=np.float64, copy=True)

    # Lazy import to keep dependencies optional
    try:
        from sklearn.preprocessing import MinMaxScaler
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("minmax_scale_dummy requires scikit-learn") from exc

    # Continuous scaling
    scaler = MinMaxScaler()
    if len(cat_indexes) != X_train_.shape[1]:
        not_cat_indexes = [i for i in range(X_train_.shape[1]) if i not in cat_indexes]
        scaler.fit(X_train_[:, not_cat_indexes])
        X_train_[:, not_cat_indexes] = scaler.transform(X_train_[:, not_cat_indexes])
        X_test_[:, not_cat_indexes] = scaler.transform(X_test_[:, not_cat_indexes])

    # One-hot categorical variables (>=3 categories)
    df_names_before = None
    df_names_after = None
    n = X_train.shape[0]
    if len(cat_indexes) > 0:
        X_train_test, df_names_before, df_names_after = dummify(
            np.concatenate((X_train_, X_test_), axis=0),
            cat_indexes=cat_indexes,
            divide_by=divide_by,
        )
        X_train_ = X_train_test[0:n, :]
        X_test_ = X_train_test[n:, :]

    if mask is not None:
        if len(cat_indexes) == 0:
            return X_train_, X_test_, mask, scaler, df_names_before, df_names_after
        mask_new = np.zeros(X_train_.shape)
        for i, var_name in enumerate(df_names_after or []):
            if "_" in var_name:
                var_ind = int(var_name.split("_")[0])
            else:
                var_ind = int(var_name)
            mask_new[:, i] = mask[:, var_ind]
        return X_train_, X_test_, mask_new, scaler, df_names_before, df_names_after

    return X_train_, X_test_, scaler, df_names_before, df_names_after


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
) -> np.ndarray:
    day_dist = _cyc_dist(a_day, b_day, period=7)
    hour_dist = _cyc_dist(a_hour, b_hour, period=24)
    time_dist = day_dist + hour_dist

    pickup_mismatch = (a_pickup[:, None] != b_pickup[None, :]).astype(np.float32)
    dropoff_mismatch = (a_dropoff[:, None] != b_dropoff[None, :]).astype(np.float32)
    space_dist = pickup_mismatch + dropoff_mismatch

    return (w_time * time_dist + w_space * space_dist).astype(np.float32)


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
) -> float:
    if ref_df.empty or other_df.empty:
        return 0.0

    ref_df = _sample_df(ref_df, max_samples, seed)
    other_df = _sample_df(other_df, max_samples, seed + 1)

    ref_day = ref_df["dia_da_semana"].to_numpy(dtype=np.int64)
    ref_hour = ref_df["hora_do_dia"].to_numpy(dtype=np.int64)
    ref_pickup = ref_df["pickup_id"].to_numpy(dtype=np.int64)
    ref_dropoff = ref_df["dropoff_id"].to_numpy(dtype=np.int64)

    other_day = other_df["dia_da_semana"].to_numpy(dtype=np.int64)
    other_hour = other_df["hora_do_dia"].to_numpy(dtype=np.int64)
    other_pickup = other_df["pickup_id"].to_numpy(dtype=np.int64)
    other_dropoff = other_df["dropoff_id"].to_numpy(dtype=np.int64)

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
        )
        min_dist = dist.min(axis=1)
        covered += int((min_dist <= radii[start:end]).sum())

    return float(100.0 * covered / max(1, n_ref))


# --- Paper-aligned metrics (PRDC coverage + EMD L1) ---
def compute_pairwise_distance(
    data_x: np.ndarray, data_y: np.ndarray | None = None
) -> np.ndarray:
    try:
        from sklearn.metrics import pairwise_distances
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("compute_pairwise_distance requires scikit-learn") from exc
    if data_y is None:
        data_y = data_x
    return pairwise_distances(data_x, data_y, metric="cityblock", n_jobs=-1)


def _get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    return k_smallests.max(axis=axis)


def _nearest_neighbour_distances(
    input_features: np.ndarray, nearest_k: int
) -> np.ndarray:
    distances = compute_pairwise_distance(input_features)
    radii = _get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def _get_prdc_runtime_config() -> Tuple[bool, str, int]:
    use_torch = True
    device = "cuda"
    chunk_size = 2048
    try:
        import config as runtime_config  # local import to avoid hard dependency

        use_torch = bool(getattr(runtime_config, "PRDC_USE_TORCH", use_torch))
        device = str(getattr(runtime_config, "PRDC_DEVICE", device))
        chunk_size = int(getattr(runtime_config, "PRDC_TORCH_CHUNK_SIZE", chunk_size))
    except Exception:
        pass
    return use_torch, device, max(1, chunk_size)


def _select_auto_k(
    real_features: np.ndarray, *, target_coverage: float = 0.95
) -> Tuple[int, float]:
    n_real = int(real_features.shape[0])
    if n_real < 2:
        return 0, 0.0

    _, inverse, counts = np.unique(
        real_features, axis=0, return_inverse=True, return_counts=True
    )
    duplicate_size_per_row = counts[inverse]
    max_k = n_real - 1

    for k in range(1, max_k + 1):
        cov_rr = float(np.mean(duplicate_size_per_row <= k))
        if cov_rr >= target_coverage:
            return k, cov_rr

    return max_k, float(np.mean(duplicate_size_per_row <= max_k))


def _coverage_prdc_sklearn(
    real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int
) -> float:
    real_nearest = _nearest_neighbour_distances(real_features, nearest_k)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)
    coverage = (distance_real_fake.min(axis=1) < real_nearest).mean()
    return float(coverage)


def _resolve_torch_device(requested_device: str):
    import torch

    requested_device = (requested_device or "cuda").strip()
    try:
        device = torch.device(requested_device)
    except Exception:
        warnings.warn(
            f"[PRDC] Invalid PRDC_DEVICE='{requested_device}'. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "[PRDC] CUDA requested for PRDC but not available. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


def _torch_chunked_kth_radii(
    features: "torch.Tensor", nearest_k: int, chunk_size: int
) -> "torch.Tensor":
    import torch

    n_samples = int(features.shape[0])
    nearest_k = int(min(max(1, nearest_k), n_samples - 1))
    radii = torch.empty((n_samples,), dtype=features.dtype, device=features.device)
    n_row_chunks = max(1, (n_samples + chunk_size - 1) // chunk_size)
    n_col_chunks = n_row_chunks
    progress_every = max(1, n_row_chunks // 10)
    print(
        f"[PRDC] real-real knn radii: n={n_samples} k={nearest_k} "
        f"row_chunks={n_row_chunks} col_chunks={n_col_chunks}"
    )

    row_chunk_idx = 0
    for row_start in range(0, n_samples, chunk_size):
        row_chunk_idx += 1
        row_end = min(n_samples, row_start + chunk_size)
        query = features[row_start:row_end]
        best = torch.full(
            (row_end - row_start, nearest_k),
            float("inf"),
            dtype=features.dtype,
            device=features.device,
        )

        for col_start in range(0, n_samples, chunk_size):
            col_end = min(n_samples, col_start + chunk_size)
            ref = features[col_start:col_end]

            dist_block = torch.cdist(query, ref, p=1)

            if col_start < row_end and col_end > row_start:
                overlap_start = max(row_start, col_start)
                overlap_end = min(row_end, col_end)
                diag = torch.arange(overlap_start, overlap_end, device=features.device)
                dist_block[diag - row_start, diag - col_start] = float("inf")

            merged = torch.cat((best, dist_block), dim=1)
            best = torch.topk(merged, k=nearest_k, dim=1, largest=False).values

        radii[row_start:row_end] = best[:, -1]
        if (
            row_chunk_idx == 1
            or row_chunk_idx == n_row_chunks
            or row_chunk_idx % progress_every == 0
        ):
            print(
                f"[PRDC] real-real knn radii progress: "
                f"{row_chunk_idx}/{n_row_chunks} rows"
            )

    return radii


def _torch_chunked_min_distances(
    ref_features: "torch.Tensor", other_features: "torch.Tensor", chunk_size: int
) -> "torch.Tensor":
    import torch

    n_ref = int(ref_features.shape[0])
    n_other = int(other_features.shape[0])
    min_dist = torch.empty((n_ref,), dtype=ref_features.dtype, device=ref_features.device)
    n_row_chunks = max(1, (n_ref + chunk_size - 1) // chunk_size)
    n_col_chunks = max(1, (n_other + chunk_size - 1) // chunk_size)
    progress_every = max(1, n_row_chunks // 10)
    print(
        f"[PRDC] real-fake min-dist: n_real={n_ref} n_fake={n_other} "
        f"row_chunks={n_row_chunks} col_chunks={n_col_chunks}"
    )

    row_chunk_idx = 0
    for row_start in range(0, n_ref, chunk_size):
        row_chunk_idx += 1
        row_end = min(n_ref, row_start + chunk_size)
        query = ref_features[row_start:row_end]
        query_min = torch.full(
            (row_end - row_start,),
            float("inf"),
            dtype=ref_features.dtype,
            device=ref_features.device,
        )

        for col_start in range(0, n_other, chunk_size):
            col_end = min(n_other, col_start + chunk_size)
            ref = other_features[col_start:col_end]
            dist_block = torch.cdist(query, ref, p=1)
            query_min = torch.minimum(query_min, dist_block.min(dim=1).values)

        min_dist[row_start:row_end] = query_min
        if (
            row_chunk_idx == 1
            or row_chunk_idx == n_row_chunks
            or row_chunk_idx % progress_every == 0
        ):
            print(
                f"[PRDC] real-fake min-dist progress: "
                f"{row_chunk_idx}/{n_row_chunks} rows"
            )

    return min_dist


def _coverage_prdc_torch(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    nearest_k: int,
    *,
    device: str,
    chunk_size: int,
) -> Tuple[float, str]:
    import torch

    torch_device = _resolve_torch_device(device)
    real_t = torch.as_tensor(real_features, dtype=torch.float32, device=torch_device)
    fake_t = torch.as_tensor(fake_features, dtype=torch.float32, device=torch_device)

    real_nearest = _torch_chunked_kth_radii(real_t, nearest_k=nearest_k, chunk_size=chunk_size)
    distance_real_fake = _torch_chunked_min_distances(
        real_t, fake_t, chunk_size=chunk_size
    )
    coverage = float((distance_real_fake < real_nearest).float().mean().item())

    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)

    return coverage, str(torch_device)


def compute_coverage_prdc(
    real_features: np.ndarray, fake_features: np.ndarray, nearest_k: int | None = None
) -> float:
    real_features = np.asarray(real_features)
    fake_features = np.asarray(fake_features)

    if real_features.size == 0 or fake_features.size == 0:
        return 0.0

    n_real = int(real_features.shape[0])
    if n_real < 2:
        return 0.0

    use_torch, requested_device, chunk_size = _get_prdc_runtime_config()
    started = time.perf_counter()

    auto_k = nearest_k is None
    if auto_k:
        auto_k_features = real_features.astype(np.float32, copy=False) if use_torch else real_features
        nearest_k, coverage_rr = _select_auto_k(auto_k_features, target_coverage=0.95)
    else:
        nearest_k = int(nearest_k)
        coverage_rr = float("nan")

    nearest_k = int(min(max(1, nearest_k), n_real - 1))

    backend = "sklearn"
    device_used = "cpu"
    try:
        if use_torch:
            coverage, device_used = _coverage_prdc_torch(
                real_features,
                fake_features,
                nearest_k,
                device=requested_device,
                chunk_size=chunk_size,
            )
            backend = "torch"
        else:
            coverage = _coverage_prdc_sklearn(real_features, fake_features, nearest_k)
    except Exception as exc:
        if use_torch:
            backend_name = "CUDA" if requested_device.lower().startswith("cuda") else "torch"
            warnings.warn(
                f"[PRDC] {backend_name} backend failed ({exc}). Falling back to sklearn CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            coverage = _coverage_prdc_sklearn(real_features, fake_features, nearest_k)
            backend = "sklearn"
            device_used = "cpu"
        else:
            raise

    elapsed_s = time.perf_counter() - started
    if auto_k:
        print(
            f"[PRDC] backend={backend} device={device_used} chunk_size={chunk_size} "
            f"auto_k={nearest_k} coverage_rr={coverage_rr:.4f} time_s={elapsed_s:.3f}"
        )
    else:
        print(
            f"[PRDC] backend={backend} device={device_used} chunk_size={chunk_size} "
            f"k={nearest_k} time_s={elapsed_s:.3f}"
        )

    return float(coverage)


def emd_l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    try:
        import ot as pot
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("emd_l1_distance requires POT (pip install POT)") from exc
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(
        pot.emd2(
            pot.unif(a.shape[0]),
            pot.unif(b.shape[0]),
            M=pot.dist(a, b, metric="cityblock"),
            numItermax = 1000000
        )
    )
