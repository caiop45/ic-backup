from __future__ import annotations

"""Downstream fare prediction evaluation utilities.

JSON schema (top-level):
{
  "skipped_fare": bool,
  "skip_reason": str | None,
  "eval_split": "val" | "hold",
  "n_train_real": float,
  "n_eval_real": float,
  "n_synth": float,
  "dwn_fare_tr_tr": {"r2": float, "r2_x100": float, "mae": float, "rmse": float},
  "dwn_fare_tr_te": {...},
  "dwn_fare_tr_syn": {...},
  "dwn_fare_syn_tr": {...},
  "dwn_fare_syn_te": {...},
  "dwn_fare_syn_syn": {...}
}
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import numpy as np
import pandas as pd

import config

try:  # pragma: no cover - exercised via tests that import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - import fallback
    _SKLEARN_AVAILABLE = False


BASE_FEATURE_COLS: tuple[str, ...] = (
    "hora_do_dia",
    "dia_da_semana",
    "pickup_id",
    "dropoff_id",
)
OPTIONAL_FEATURE_COLS: tuple[str, ...] = ("r", "passenger_count")
LABEL_COL = "total_amount"


@dataclass(frozen=True)
class FareEvalInputs:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    synth_df: pd.DataFrame
    eval_split: str


def _subsample_df(df: pd.DataFrame, n_rows: int | None, seed: int) -> pd.DataFrame:
    if n_rows is None or n_rows <= 0 or len(df) <= n_rows:
        return df.reset_index(drop=True)
    return df.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def _select_feature_cols(
    train_df: pd.DataFrame, eval_df: pd.DataFrame, synth_df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    required = list(BASE_FEATURE_COLS)
    missing: list[str] = []
    for col in required:
        if col not in train_df.columns or col not in eval_df.columns or col not in synth_df.columns:
            missing.append(col)
    if missing:
        return [], missing

    features = required.copy()
    for col in OPTIONAL_FEATURE_COLS:
        if (
            col in train_df.columns
            and col in eval_df.columns
            and col in synth_df.columns
        ):
            features.append(col)
    return features, []


def _clean_df(
    df: pd.DataFrame, feature_cols: Sequence[str], label_col: str
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce").astype("float64")
    for col in feature_cols:
        if col == "r":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    mask = out[label_col].notna()
    for col in feature_cols:
        mask &= out[col].notna()
    return out.loc[mask].reset_index(drop=True)


def _build_model(model_name: str, seed: int) -> object:
    name = model_name.lower().strip()
    if name == "gbr":
        return GradientBoostingRegressor(random_state=seed)
    if name == "hgbr":
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "linear":
        return LinearRegression()
    raise ValueError("model must be one of: gbr, hgbr, linear")


def _build_pipeline(
    *, cat_cols: Sequence[str], num_cols: Sequence[str], model_name: str, seed: int
) -> Pipeline:
    transformers = []
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(cat_cols))
        )
    if num_cols:
        transformers.append(("num", StandardScaler(), list(num_cols)))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = _build_model(model_name, seed)
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2),
        "r2_x100": float(r2 * 100.0),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def _fit_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    label_col: str,
    model_name: str,
    seed: int,
) -> dict[str, float]:
    num_cols = [col for col in feature_cols if col == "r"]
    cat_cols = [col for col in feature_cols if col != "r"]
    pipeline = _build_pipeline(
        cat_cols=cat_cols, num_cols=num_cols, model_name=model_name, seed=seed
    )
    X_train = train_df[list(feature_cols)]
    y_train = train_df[label_col].to_numpy(dtype=np.float64)
    X_test = test_df[list(feature_cols)]
    y_test = test_df[label_col].to_numpy(dtype=np.float64)

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return _regression_metrics(y_test, preds)


def compute_downstream_fare_metrics(
    inputs: FareEvalInputs,
    *,
    train_rows: int | None = None,
    test_rows: int | None = None,
    model_name: str = "gbr",
    seed: int | None = None,
    label_col: str = LABEL_COL,
) -> dict[str, object]:
    """Compute downstream fare regression metrics with real/synth train/test combos."""
    if not _SKLEARN_AVAILABLE:
        return {
            "skipped_fare": True,
            "skip_reason": "sklearn_missing",
            "eval_split": inputs.eval_split,
        }

    if seed is None:
        seed = int(getattr(config, "GLOBAL_SEED", 0))

    feature_cols, missing = _select_feature_cols(
        inputs.train_df, inputs.eval_df, inputs.synth_df
    )
    if label_col not in inputs.train_df.columns or label_col not in inputs.eval_df.columns:
        missing.append(label_col)
    if label_col not in inputs.synth_df.columns:
        missing.append(label_col)
    if missing:
        return {
            "skipped_fare": True,
            "skip_reason": f"missing_columns:{','.join(sorted(set(missing)))}",
            "eval_split": inputs.eval_split,
        }

    train_df = _clean_df(inputs.train_df, feature_cols, label_col)
    eval_df = _clean_df(inputs.eval_df, feature_cols, label_col)
    synth_df = _clean_df(inputs.synth_df, feature_cols, label_col)

    if len(train_df) == 0 or len(eval_df) == 0 or len(synth_df) == 0:
        return {
            "skipped_fare": True,
            "skip_reason": "empty_after_filter",
            "eval_split": inputs.eval_split,
        }

    train_df = _subsample_df(train_df, train_rows, seed)
    eval_df = _subsample_df(eval_df, test_rows, seed + 1)
    synth_df = _subsample_df(synth_df, test_rows, seed + 2)

    if len(train_df) < 2 or len(eval_df) < 2 or len(synth_df) < 2:
        return {
            "skipped_fare": True,
            "skip_reason": "insufficient_rows",
            "eval_split": inputs.eval_split,
        }

    metrics: dict[str, object] = {
        "skipped_fare": False,
        "skip_reason": None,
        "eval_split": inputs.eval_split,
        "n_train_real": float(len(train_df)),
        "n_eval_real": float(len(eval_df)),
        "n_synth": float(len(synth_df)),
    }

    metrics["dwn_fare_tr_tr"] = _fit_and_eval(
        train_df,
        train_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    metrics["dwn_fare_tr_te"] = _fit_and_eval(
        train_df,
        eval_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    metrics["dwn_fare_tr_syn"] = _fit_and_eval(
        train_df,
        synth_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    metrics["dwn_fare_syn_tr"] = _fit_and_eval(
        synth_df,
        train_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    metrics["dwn_fare_syn_te"] = _fit_and_eval(
        synth_df,
        eval_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    metrics["dwn_fare_syn_syn"] = _fit_and_eval(
        synth_df,
        synth_df,
        feature_cols=feature_cols,
        label_col=label_col,
        model_name=model_name,
        seed=seed,
    )
    return metrics
