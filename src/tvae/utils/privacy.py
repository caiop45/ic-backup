from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import json
import numpy as np
import pandas as pd

from tvae import config
from tvae.utils.metrics import distance_matrix


@dataclass(frozen=True)
class PrivacyReport:
    exact_match_rate_discrete: float
    exact_match_rate_with_r_rounded: float | None
    min_distance_quantiles: dict[str, float]
    min_distance_summary: dict[str, float]


def _sample_df(df: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df.reset_index(drop=True)
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def _rounded_r_values(
    df: pd.DataFrame, *, r_col: str, decimals: int
) -> np.ndarray:
    scale = 10 ** int(decimals)
    values = df[r_col].to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=-1.0, posinf=1.0, neginf=-1.0)
    return np.trunc(values * scale).astype(np.int64)


def exact_match_rate(
    train_df: pd.DataFrame, synth_df: pd.DataFrame, cols: Sequence[str]
) -> float:
    """Exact match rate of synth rows against train keys (discrete cols only)."""
    if train_df.empty or synth_df.empty:
        return 0.0
    train_keys = set(train_df[list(cols)].itertuples(index=False, name=None))
    synth_keys = list(synth_df[list(cols)].itertuples(index=False, name=None))
    if not synth_keys:
        return 0.0
    matches = sum(1 for key in synth_keys if key in train_keys)
    return float(matches / len(synth_keys))


def exact_match_rate_with_r(
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    r_col: str,
    decimals: int,
) -> float | None:
    """Exact match rate including rounded residual r, if present."""
    if r_col not in train_df.columns or r_col not in synth_df.columns:
        return None
    if train_df.empty or synth_df.empty:
        return 0.0

    train_r = _rounded_r_values(train_df, r_col=r_col, decimals=decimals)
    synth_r = _rounded_r_values(synth_df, r_col=r_col, decimals=decimals)

    train_keys = []
    for idx, row in enumerate(train_df[list(cols)].itertuples(index=False, name=None)):
        train_keys.append(tuple(row) + (int(train_r[idx]),))
    synth_keys = []
    for idx, row in enumerate(synth_df[list(cols)].itertuples(index=False, name=None)):
        synth_keys.append(tuple(row) + (int(synth_r[idx]),))

    if not synth_keys:
        return 0.0
    train_set = set(train_keys)
    matches = sum(1 for key in synth_keys if key in train_set)
    return float(matches / len(synth_keys))


def min_distances_to_train(
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    max_samples: int | None = None,
    max_samples_train: int | None = None,
    max_samples_synth: int | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    w_time: float | None = None,
    w_space: float | None = None,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
) -> np.ndarray:
    """Compute min distance from each synth row to the train set."""
    if train_df.empty or synth_df.empty:
        return np.array([], dtype=np.float32)

    if seed is None:
        seed = int(getattr(config, "GLOBAL_SEED", 0))
    if max_samples is None:
        max_samples = getattr(config, "DCR_MAX_SAMPLES", None)
    if max_samples_train is None:
        max_samples_train = max_samples
    if max_samples_synth is None:
        max_samples_synth = max_samples

    train_df = _sample_df(train_df, max_samples_train, seed)
    synth_df = _sample_df(synth_df, max_samples_synth, seed + 1)

    if chunk_size is None:
        chunk_size = int(getattr(config, "DCR_CHUNK_SIZE", 1024))
    if w_time is None:
        w_time = float(getattr(config, "COVERAGE_TIME_WEIGHT", 1.0))
    if w_space is None:
        w_space = float(getattr(config, "COVERAGE_SPACE_WEIGHT", 1.0))
    if w_residual is None:
        w_residual = float(getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0))
    if use_residual is None:
        use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if residual_col is None:
        residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))

    min_dists: list[np.ndarray] = []
    for start in range(0, len(synth_df), chunk_size):
        end = min(len(synth_df), start + chunk_size)
        chunk = synth_df.iloc[start:end]
        dist = distance_matrix(
            chunk,
            train_df,
            w_time=w_time,
            w_space=w_space,
            w_residual=w_residual,
            use_residual=use_residual,
            residual_col=residual_col,
        )
        if dist.size == 0:
            chunk_min = np.full(end - start, np.inf, dtype=np.float32)
        else:
            chunk_min = dist.min(axis=1)
        min_dists.append(chunk_min)
    if not min_dists:
        return np.array([], dtype=np.float32)
    return np.concatenate(min_dists)


def min_distance_quantiles(
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    quantiles: Sequence[float] | None = None,
    **kwargs: object,
) -> dict[str, float]:
    """Compute quantiles of min distances from synth to train."""
    if quantiles is None:
        quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    dists = min_distances_to_train(train_df, synth_df, **kwargs)
    if dists.size == 0:
        return {}
    qs = np.array(list(quantiles), dtype=np.float64)
    vals = np.quantile(dists, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


def build_privacy_report(
    train_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    discrete_cols: Sequence[str],
    r_col: str,
    r_decimals: int,
    quantiles: Sequence[float] | None = None,
    max_samples: int | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    w_time: float | None = None,
    w_space: float | None = None,
    w_residual: float | None = None,
    use_residual: bool | None = None,
    residual_col: str | None = None,
) -> tuple[PrivacyReport, dict[str, float]]:
    """Build privacy report and metrics dict (match rates)."""
    rate_discrete = exact_match_rate(train_df, synth_df, discrete_cols)
    rate_with_r = exact_match_rate_with_r(
        train_df,
        synth_df,
        discrete_cols,
        r_col=r_col,
        decimals=r_decimals,
    )

    dists = min_distances_to_train(
        train_df,
        synth_df,
        max_samples=max_samples,
        seed=seed,
        chunk_size=chunk_size,
        w_time=w_time,
        w_space=w_space,
        w_residual=w_residual,
        use_residual=use_residual,
        residual_col=residual_col,
    )
    if quantiles is None:
        quantiles = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    if dists.size == 0:
        quantiles_dict = {}
        summary = {}
    else:
        qs = np.array(list(quantiles), dtype=np.float64)
        vals = np.quantile(dists, qs)
        quantiles_dict = {str(q): float(v) for q, v in zip(qs, vals)}
        summary = {
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "mean": float(np.mean(dists)),
            "median": float(np.median(dists)),
        }

    report = PrivacyReport(
        exact_match_rate_discrete=rate_discrete,
        exact_match_rate_with_r_rounded=rate_with_r,
        min_distance_quantiles=quantiles_dict,
        min_distance_summary=summary,
    )

    metrics = {"exact_match_rate_discrete": float(rate_discrete)}
    if rate_with_r is not None:
        metrics["exact_match_rate_with_r_rounded"] = float(rate_with_r)
    return report, metrics


def save_privacy_report(report: PrivacyReport, path: str | Path, **meta: object) -> None:
    payload = {
        "exact_match_rate_discrete": report.exact_match_rate_discrete,
        "exact_match_rate_with_r_rounded": report.exact_match_rate_with_r_rounded,
        "min_distance_quantiles": report.min_distance_quantiles,
        "min_distance_summary": report.min_distance_summary,
        "meta": meta,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def rejection_filter(
    train_df: pd.DataFrame,
    *,
    generator_fn: Callable[[int, int | None], pd.DataFrame],
    target_rows: int,
    tau: float,
    max_iters: int,
    seed: int | None = None,
    **distance_kwargs: object,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Rejection sampling: keep rows with min distance >= tau."""
    if target_rows <= 0:
        return pd.DataFrame(), {"accepted": 0.0, "generated": 0.0}
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if train_df.empty:
        candidates = generator_fn(target_rows, seed)
        stats = {
            "accepted": float(len(candidates)),
            "generated": float(len(candidates)),
            "accept_rate": 1.0 if len(candidates) > 0 else 0.0,
            "tau": float(tau),
            "max_iters": float(max_iters),
        }
        return candidates, stats

    accepted: list[pd.DataFrame] = []
    total_generated = 0
    for i in range(int(max_iters)):
        remaining = target_rows - sum(len(df) for df in accepted)
        if remaining <= 0:
            break
        batch_seed = None if seed is None else int(seed) + i
        candidates = generator_fn(remaining, batch_seed)
        if candidates.empty:
            break
        total_generated += len(candidates)
        distances = min_distances_to_train(train_df, candidates, seed=batch_seed, **distance_kwargs)
        if distances.size == 0:
            keep_mask = np.zeros(len(candidates), dtype=bool)
        else:
            keep_mask = distances >= float(tau)
        if keep_mask.any():
            kept = candidates.loc[keep_mask].reset_index(drop=True)
            accepted.append(kept)

    if accepted:
        out_df = pd.concat(accepted, ignore_index=True)
    else:
        out_df = pd.DataFrame()
    if len(out_df) > target_rows:
        out_df = out_df.iloc[:target_rows].reset_index(drop=True)

    stats = {
        "accepted": float(len(out_df)),
        "generated": float(total_generated),
        "accept_rate": float(len(out_df) / max(1, total_generated)),
        "tau": float(tau),
        "max_iters": float(max_iters),
    }
    return out_df, stats
