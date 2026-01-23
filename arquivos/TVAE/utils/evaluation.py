"""Shared evaluation metrics for TVAE and Strategy A."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

import config
from utils.metrics import (
    chi2_counts,
    coverage_score,
    graph_similarity_score,
    joint_metrics,
    jsd_counts,
    marginal_counts,
    od_metrics,
    plot_marginal_hist,
    plot_topk,
    time_metrics,
    topk_table,
    wasserstein_time,
)


def compute_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    order_key: str,
    output_dir: Path,
    plot_dir: Path,
) -> Dict[str, float]:
    """Compute pipeline metrics (marginals + OD/time/joint) and plots.

    Output keys and behavior match the previous train_tvae.py implementation.
    """
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
    return metrics


def compute_paper_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, synth_df: pd.DataFrame
) -> Dict[str, float]:
    """Compute paper metrics (W1 time, graph similarity, coverage)."""
    metrics: Dict[str, float] = {}

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
    )

    metrics["coverage_k"] = float(config.COVERAGE_K)
    metrics["coverage_max_samples"] = float(config.COVERAGE_MAX_SAMPLES or 0)
    metrics["coverage_time_weight"] = float(config.COVERAGE_TIME_WEIGHT)
    metrics["coverage_space_weight"] = float(config.COVERAGE_SPACE_WEIGHT)

    return metrics


__all__ = ["compute_metrics", "compute_paper_metrics"]
