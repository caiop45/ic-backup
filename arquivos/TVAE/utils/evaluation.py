"""Shared evaluation metrics for TVAE and THT-TripGen."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import numpy as np

import config
from utils.metrics import (
    chi2_counts,
    coverage_score,
    dcr_quantile,
    dcr_within_quantile,
    graph_similarity_score,
    joint_metrics,
    jsd_counts,
    marginal_counts,
    median_profile_mae,
    od_metrics,
    plot_marginal_hist,
    plot_continuous_hist,
    plot_topk,
    quantile_mae,
    safe_log1p,
    time_metrics,
    topk_table,
    wasserstein_time,
    wasserstein_1d_continuous,
    ks_statistic_1d,
)


def _downsample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Deterministically downsample a dataframe to n rows (no replacement)."""
    if n <= 0:
        return df.head(0).copy()
    if len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, replace=False, random_state=seed).reset_index(drop=True)


def _bootstrap_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a bootstrap sample (with replacement) of size n."""
    if n <= 0:
        return df.head(0).copy()
    return df.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)


def _suffix_metrics(metrics: Dict[str, float], suffix: str) -> Dict[str, float]:
    return {f"{key}{suffix}": float(value) for key, value in metrics.items()}


def _alpha_key(alpha: float) -> str:
    if alpha < 0.01:
        return f"p{int(round(alpha * 1000)):03d}"
    return f"p{int(round(alpha * 100)):02d}"


def _filter_non_null(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col in cols:
        if col in df.columns:
            mask &= df[col].notna()
    return df.loc[mask].reset_index(drop=True)


def _fare_scale(
    train_df: pd.DataFrame, *, fare_col: str, method: str, eps: float
) -> float:
    if fare_col not in train_df.columns:
        return 1.0
    values = safe_log1p(train_df[fare_col])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    method = method.lower().strip()
    if method == "iqr":
        q75, q25 = np.quantile(values, [0.75, 0.25])
        scale = float(q75 - q25)
        if scale <= eps:
            scale = float(np.std(values))
    elif method == "std":
        scale = float(np.std(values))
    else:
        raise ValueError("fare scale method must be 'iqr' or 'std'")
    if not np.isfinite(scale) or scale <= eps:
        return 1.0
    return float(scale)


def _pax_scale(train_df: pd.DataFrame, *, pax_col: str, eps: float) -> float:
    if pax_col not in train_df.columns:
        return 1.0
    values = pd.to_numeric(train_df[pax_col], errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    scale = float(np.max(values) - np.min(values))
    if not np.isfinite(scale) or scale <= eps:
        return 1.0
    return float(scale)


def compute_attribute_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    fare_col: str | None = None,
    passenger_col: str | None = None,
    fare_cols: Sequence[str] | None = None,
    passenger_cols: Sequence[str] | None = None,
    quantiles: Sequence[float] | None = None,
    plot_dir: Path | None = None,
    order_key: str | None = None,
) -> Dict[str, float]:
    """Compute marginal/conditional metrics for discrete/continuous attributes."""
    if passenger_cols is None:
        if passenger_col is not None:
            passenger_cols = [passenger_col]
        else:
            passenger_cols = list(
                getattr(
                    config,
                    "THT_ATTRIBUTE_DISCRETE_COLUMNS",
                    [str(getattr(config, "PASSENGER_COL", "passenger_count"))],
                )
            )

    if fare_cols is None:
        if fare_col is not None:
            fare_cols = [fare_col]
        else:
            fare_cols = list(
                getattr(
                    config,
                    "THT_ATTRIBUTE_CONTINUOUS_COLUMNS",
                    [str(getattr(config, "FARE_COL", "total_amount"))],
                )
            )

    # keep deterministic ordering and avoid duplicates if config contains repeats
    def _dedup(columns: Sequence[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for col in columns:
            if col in seen:
                continue
            out.append(col)
            seen.add(col)
        return out

    passenger_cols = _dedup(passenger_cols)
    fare_cols = _dedup(fare_cols)
    if quantiles is None:
        quantiles = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]

    metrics: Dict[str, float] = {}

    for passenger_col in passenger_cols:
        if passenger_col in real_df.columns and passenger_col in synth_df.columns:
            real_counts = marginal_counts(real_df, passenger_col)
            synth_counts = marginal_counts(synth_df, passenger_col)
            metrics[f"{passenger_col}_jsd"] = jsd_counts(real_counts, synth_counts)
            metrics[f"{passenger_col}_chi2"] = chi2_counts(real_counts, synth_counts)
            if plot_dir is not None and order_key is not None:
                plot_marginal_hist(
                    real_counts,
                    synth_counts,
                    title=f"{order_key} {passenger_col}",
                    path=plot_dir / f"hist_{order_key}_{passenger_col}.png",
                )

    for fare_col in fare_cols:
        if fare_col in real_df.columns and fare_col in synth_df.columns:
            real_z = safe_log1p(real_df[fare_col])
            synth_z = safe_log1p(synth_df[fare_col])
            metrics[f"{fare_col}_log1p_w1"] = wasserstein_1d_continuous(real_z, synth_z)
            metrics[f"{fare_col}_log1p_ks"] = ks_statistic_1d(real_z, synth_z)
            metrics[f"{fare_col}_log1p_quantile_mae"] = quantile_mae(
                real_z, synth_z, quantiles
            )

            if "hour_of_day" in real_df.columns and "hour_of_day" in synth_df.columns:
                real_tmp = pd.DataFrame(
                    {"hour_of_day": real_df["hour_of_day"], "_val": real_z}
                )
                synth_tmp = pd.DataFrame(
                    {"hour_of_day": synth_df["hour_of_day"], "_val": synth_z}
                )
                metrics[f"{fare_col}_log1p_median_mae_by_hour"] = median_profile_mae(
                    real_tmp, synth_tmp, group_col="hour_of_day", value_col="_val"
                )

            for passenger_col in passenger_cols:
                if passenger_col in real_df.columns and passenger_col in synth_df.columns:
                    real_tmp = pd.DataFrame(
                        {passenger_col: real_df[passenger_col], "_val": real_z}
                    )
                    synth_tmp = pd.DataFrame(
                        {passenger_col: synth_df[passenger_col], "_val": synth_z}
                    )
                    metrics[f"{fare_col}_log1p_median_mae_by_{passenger_col}"] = (
                        median_profile_mae(
                            real_tmp, synth_tmp, group_col=passenger_col, value_col="_val"
                        )
                    )

            if plot_dir is not None and order_key is not None:
                plot_continuous_hist(
                    real_z,
                    synth_z,
                    title=f"{order_key} {fare_col} (log1p)",
                    path=plot_dir / f"hist_{order_key}_{fare_col}_log1p.png",
                )


    return metrics


def compute_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    order_key: str,
    output_dir: Path,
    plot_dir: Path,
    compute_rr_baselines: bool | None = None,
    random_state: int | None = None,
) -> Dict[str, float]:
    """Compute pipeline metrics (marginals + OD/time/joint) and plots.

    Includes real-vs-real (RR) baselines for OD/joint metrics to contextualize
    coverage and divergence under finite sample sizes.
    Output keys and behavior match the previous train_tvae.py implementation.
    """
    if compute_rr_baselines is None:
        compute_rr_baselines = bool(
            getattr(config, "METRICS_COMPUTE_RR_BASELINES", True)
        )
    if random_state is None:
        random_state = int(
            getattr(config, "METRICS_RR_BOOTSTRAP_SEED", config.GLOBAL_SEED)
        )

    n_samples = int(min(len(real_df), len(synth_df)))
    real_df = _downsample_df(real_df, n_samples, random_state)
    synth_df = _downsample_df(synth_df, n_samples, random_state)

    metrics: Dict[str, float] = {}
    metrics["n_real"] = float(len(real_df))
    metrics["n_synth"] = float(len(synth_df))

    for col in config.OUTPUT_COLUMNS:
        real_counts = marginal_counts(real_df, col)
        synth_counts = marginal_counts(synth_df, col)
        metrics[f"{col}_jsd"] = jsd_counts(real_counts, synth_counts)
        metrics[f"{col}_chi2"] = chi2_counts(real_counts, synth_counts)

        plot_path = plot_dir / f"hist_{order_key}_{col}.png"
        plot_marginal_hist(real_counts, synth_counts, title=f"{order_key} {col}", path=plot_path)

        if col in ("pickup_id", "dropoff_id"):
            topk_df = topk_table(real_counts, synth_counts, k=20)
            topk_path = output_dir / f"topk_{order_key}_{col}.csv"
            topk_df.to_csv(topk_path, index=False)
            plot_topk(
                topk_df,
                title=f"{order_key} topk {col}",
                path=plot_dir / f"topk_{order_key}_{col}.png",
            )

    od = od_metrics(real_df, synth_df)
    metrics.update(od)

    time = time_metrics(real_df, synth_df)
    metrics.update(time)

    joint = joint_metrics(real_df, synth_df)
    metrics.update(joint)

    if compute_rr_baselines:
        rr_a = _bootstrap_df(real_df, n_samples, random_state)
        rr_b = _bootstrap_df(real_df, n_samples, random_state + 1)
        metrics.update(_suffix_metrics(od_metrics(rr_a, rr_b), "_rr"))
        metrics.update(_suffix_metrics(joint_metrics(rr_a, rr_b), "_rr"))
    return metrics


def compute_paper_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    include_within: bool = False,
    use_residual_distance: bool | None = None,
    residual_col: str | None = None,
    coverage_residual_weight: float | None = None,
    dcr_residual_weight: float | None = None,
    enable_extended_privacy: bool | None = None,
    fare_col: str | None = None,
    passenger_col: str | None = None,
) -> Dict[str, float]:
    """Compute paper metrics (W1 time, graph similarity, coverage, DCR/rDCR).

    When include_within=True, also compute within-set DCR (real-real, synth-synth)
    excluding self-matches.
    Residual time r is included in distance metrics when enabled.
    """
    metrics: Dict[str, float] = {}

    if use_residual_distance is None:
        use_residual_distance = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    if coverage_residual_weight is None:
        coverage_residual_weight = float(getattr(config, "COVERAGE_RESIDUAL_WEIGHT", 1.0))
    if dcr_residual_weight is None:
        dcr_residual_weight = float(getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0))
    if enable_extended_privacy is None:
        enable_extended_privacy = bool(
            getattr(config, "EVAL_ENABLE_EXTENDED_PRIVACY", True)
        )
    if fare_col is None:
        fare_col = str(getattr(config, "FARE_COL", "total_amount"))
    if passenger_col is None:
        passenger_col = str(getattr(config, "PASSENGER_COL", "passenger_count"))

    metrics["w1_tr_te"] = wasserstein_time(
        train_df, test_df, support_size=config.TIME_KEY_CARDINALITY
    )
    metrics["w1_tr_syn"] = wasserstein_time(
        train_df, synth_df, support_size=config.TIME_KEY_CARDINALITY
    )
    metrics["w1_te_syn"] = wasserstein_time(
        test_df, synth_df, support_size=config.TIME_KEY_CARDINALITY
    )

    metrics["g_tr_te"] = 100.0 * graph_similarity_score(train_df, test_df)
    metrics["g_tr_syn"] = 100.0 * graph_similarity_score(train_df, synth_df)
    metrics["g_te_syn"] = 100.0 * graph_similarity_score(test_df, synth_df)

    metrics["cov_tr_te"] = coverage_score(
        train_df,
        test_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
        w_residual=coverage_residual_weight,
        use_residual=use_residual_distance,
        residual_col=residual_col,
    )
    metrics["cov_tr_syn"] = coverage_score(
        train_df,
        synth_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
        w_residual=coverage_residual_weight,
        use_residual=use_residual_distance,
        residual_col=residual_col,
    )
    metrics["cov_te_syn"] = coverage_score(
        test_df,
        synth_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
        w_residual=coverage_residual_weight,
        use_residual=use_residual_distance,
        residual_col=residual_col,
    )

    metrics["coverage_k"] = float(config.COVERAGE_K)
    metrics["coverage_max_samples"] = float(config.COVERAGE_MAX_SAMPLES or 0)
    metrics["coverage_time_weight"] = float(config.COVERAGE_TIME_WEIGHT)
    metrics["coverage_space_weight"] = float(config.COVERAGE_SPACE_WEIGHT)

    dcr_alpha = float(getattr(config, "DCR_ALPHA", 0.05))
    dcr_max_samples = getattr(config, "DCR_MAX_SAMPLES", None)
    dcr_chunk = int(getattr(config, "DCR_CHUNK_SIZE", 1024))
    dcr_eps = float(getattr(config, "DCR_EPS", 1e-12))

    dcr_tr_syn = dcr_quantile(
        train_df,
        synth_df,
        alpha=dcr_alpha,
        max_samples=dcr_max_samples,
        chunk_size=dcr_chunk,
        seed=config.GLOBAL_SEED,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
        w_residual=dcr_residual_weight,
        use_residual=use_residual_distance,
        residual_col=residual_col,
    )
    dcr_hold_syn = dcr_quantile(
        test_df,
        synth_df,
        alpha=dcr_alpha,
        max_samples=dcr_max_samples,
        chunk_size=dcr_chunk,
        seed=config.GLOBAL_SEED,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
        w_residual=dcr_residual_weight,
        use_residual=use_residual_distance,
        residual_col=residual_col,
    )
    metrics["dcr_tr_syn_p05"] = float(dcr_tr_syn)
    metrics["dcr_hold_syn_p05"] = float(dcr_hold_syn)
    metrics["rdcr_p05"] = float(dcr_tr_syn / max(dcr_hold_syn, dcr_eps))
    metrics["dcr_alpha"] = float(dcr_alpha)
    metrics["dcr_max_samples"] = float(dcr_max_samples or 0)
    metrics["dcr_chunk_size"] = float(dcr_chunk)
    metrics["dcr_within_excludes_self"] = 1.0

    if include_within:
        dcr_rr = dcr_within_quantile(
            train_df,
            alpha=dcr_alpha,
            max_samples=dcr_max_samples,
            chunk_size=dcr_chunk,
            seed=config.GLOBAL_SEED,
            w_time=config.COVERAGE_TIME_WEIGHT,
            w_space=config.COVERAGE_SPACE_WEIGHT,
            w_residual=dcr_residual_weight,
            use_residual=use_residual_distance,
            residual_col=residual_col,
        )
        dcr_ss = dcr_within_quantile(
            synth_df,
            alpha=dcr_alpha,
            max_samples=dcr_max_samples,
            chunk_size=dcr_chunk,
            seed=config.GLOBAL_SEED,
            w_time=config.COVERAGE_TIME_WEIGHT,
            w_space=config.COVERAGE_SPACE_WEIGHT,
            w_residual=dcr_residual_weight,
            use_residual=use_residual_distance,
            residual_col=residual_col,
        )
        metrics["dcr_rr_p05"] = float(dcr_rr)
        metrics["dcr_ss_p05"] = float(dcr_ss)

    if enable_extended_privacy:
        use_pax = (
            passenger_col in train_df.columns
            and passenger_col in test_df.columns
            and passenger_col in synth_df.columns
        )
        use_fare = (
            fare_col in train_df.columns
            and fare_col in test_df.columns
            and fare_col in synth_df.columns
        )
        use_residual_ext = (
            use_residual_distance
            and residual_col in train_df.columns
            and residual_col in test_df.columns
            and residual_col in synth_df.columns
        )
        if use_pax or use_fare:
            ext_cols: list[str] = []
            if use_residual_ext:
                ext_cols.append(residual_col)
            if use_pax:
                ext_cols.append(passenger_col)
            if use_fare:
                ext_cols.append(fare_col)

            if ext_cols:
                scale_eps = float(getattr(config, "DCR_SCALE_EPS", 1e-6))
                fare_scale = _fare_scale(
                    train_df,
                    fare_col=fare_col,
                    method=str(getattr(config, "DCR_FARE_SCALE_METHOD", "iqr")),
                    eps=scale_eps,
                )
                pax_scale = _pax_scale(train_df, pax_col=passenger_col, eps=scale_eps)

                ext_train = _filter_non_null(train_df, ext_cols)
                ext_test = _filter_non_null(test_df, ext_cols)
                ext_synth = _filter_non_null(synth_df, ext_cols)

                if len(ext_train) > 0 and len(ext_test) > 0 and len(ext_synth) > 0:
                    w_time = float(getattr(config, "DCR_W_TIME", 1.0))
                    w_space = float(getattr(config, "DCR_W_SPACE", 1.0))
                    w_r = float(getattr(config, "DCR_W_R", 0.5))
                    w_pax = float(getattr(config, "DCR_W_PAX", 0.5))
                    w_fare = float(getattr(config, "DCR_W_FARE", 0.5))

                    metrics["cov_tr_te_ext"] = coverage_score(
                        ext_train,
                        ext_test,
                        k=config.COVERAGE_K,
                        max_samples=config.COVERAGE_MAX_SAMPLES,
                        seed=config.GLOBAL_SEED,
                        chunk_size=config.COVERAGE_CHUNK_SIZE,
                        w_time=w_time,
                        w_space=w_space,
                        w_residual=w_r,
                        use_residual=use_residual_distance,
                        residual_col=residual_col,
                        w_passenger=w_pax,
                        w_fare=w_fare,
                        passenger_col=passenger_col,
                        fare_col=fare_col,
                        pax_scale=pax_scale,
                        fare_scale=fare_scale,
                    )
                    metrics["cov_tr_syn_ext"] = coverage_score(
                        ext_train,
                        ext_synth,
                        k=config.COVERAGE_K,
                        max_samples=config.COVERAGE_MAX_SAMPLES,
                        seed=config.GLOBAL_SEED,
                        chunk_size=config.COVERAGE_CHUNK_SIZE,
                        w_time=w_time,
                        w_space=w_space,
                        w_residual=w_r,
                        use_residual=use_residual_distance,
                        residual_col=residual_col,
                        w_passenger=w_pax,
                        w_fare=w_fare,
                        passenger_col=passenger_col,
                        fare_col=fare_col,
                        pax_scale=pax_scale,
                        fare_scale=fare_scale,
                    )
                    metrics["cov_te_syn_ext"] = coverage_score(
                        ext_test,
                        ext_synth,
                        k=config.COVERAGE_K,
                        max_samples=config.COVERAGE_MAX_SAMPLES,
                        seed=config.GLOBAL_SEED,
                        chunk_size=config.COVERAGE_CHUNK_SIZE,
                        w_time=w_time,
                        w_space=w_space,
                        w_residual=w_r,
                        use_residual=use_residual_distance,
                        residual_col=residual_col,
                        w_passenger=w_pax,
                        w_fare=w_fare,
                        passenger_col=passenger_col,
                        fare_col=fare_col,
                        pax_scale=pax_scale,
                        fare_scale=fare_scale,
                    )

                    alphas = list(getattr(config, "DCR_ALPHAS", [dcr_alpha]))
                    for alpha in sorted(set(float(a) for a in alphas)):
                        key = _alpha_key(float(alpha))
                        dcr_tr = dcr_quantile(
                            ext_train,
                            ext_synth,
                            alpha=float(alpha),
                            max_samples=dcr_max_samples,
                            chunk_size=dcr_chunk,
                            seed=config.GLOBAL_SEED,
                            w_time=w_time,
                            w_space=w_space,
                            w_residual=w_r,
                            use_residual=use_residual_distance,
                            residual_col=residual_col,
                            w_passenger=w_pax,
                            w_fare=w_fare,
                            passenger_col=passenger_col,
                            fare_col=fare_col,
                            pax_scale=pax_scale,
                            fare_scale=fare_scale,
                        )
                        dcr_hold = dcr_quantile(
                            ext_test,
                            ext_synth,
                            alpha=float(alpha),
                            max_samples=dcr_max_samples,
                            chunk_size=dcr_chunk,
                            seed=config.GLOBAL_SEED,
                            w_time=w_time,
                            w_space=w_space,
                            w_residual=w_r,
                            use_residual=use_residual_distance,
                            residual_col=residual_col,
                            w_passenger=w_pax,
                            w_fare=w_fare,
                            passenger_col=passenger_col,
                            fare_col=fare_col,
                            pax_scale=pax_scale,
                            fare_scale=fare_scale,
                        )
                        metrics[f"dcr_tr_syn_{key}_ext"] = float(dcr_tr)
                        metrics[f"dcr_hold_syn_{key}_ext"] = float(dcr_hold)
                        metrics[f"rdcr_{key}_ext"] = float(
                            dcr_tr / max(dcr_hold, dcr_eps)
                        )

    return metrics


__all__ = ["compute_metrics", "compute_paper_metrics", "compute_attribute_metrics"]
