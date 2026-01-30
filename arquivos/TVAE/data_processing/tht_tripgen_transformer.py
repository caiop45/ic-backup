"""THT-TripGen transformer with shared zone vocabulary.

Schema (THT-TripGen input):
    - categorical: hora_do_dia, pickup_id, dropoff_id, dia_da_semana
      optional: is_weekend, month
    - continuous: r in [0, 1)

Outputs index-space columns:
    h_idx, o_idx, d_idx, dow_idx, (optional) is_weekend_idx, month_idx, plus r.

Pickup and dropoff share the same zone vocabulary derived from train data only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

import config

TIME_COL = "hora_do_dia"
PICKUP_COL = "pickup_id"
DROPOFF_COL = "dropoff_id"
DOW_COL = "dia_da_semana"
R_COL = "r"
CONDITIONAL_IDX_NAMES = {
    DOW_COL: "dow_idx",
    "is_weekend": "is_weekend_idx",
    "month": "month_idx",
}


@dataclass
class THTTripGenTransformerState:
    zone_categories: List[int]
    time_categories: List[int]
    conditional_categories: Dict[str, List[int]]
    conditional_columns: List[str]
    min_r_eps: float
    use_weekend: bool
    use_month: bool


class THTTripGenTransformer:
    """Transformer for THT-TripGen with a shared zone vocabulary."""

    def __init__(
        self,
        *,
        min_r_eps: float | None = None,
        use_weekend: bool | None = None,
        use_month: bool | None = None,
    ) -> None:
        self.min_r_eps = float(config.THT_MIN_R_EPS) if min_r_eps is None else float(min_r_eps)
        self.use_weekend = config.THT_USE_WEEKEND if use_weekend is None else bool(use_weekend)
        self.use_month = config.THT_USE_MONTH if use_month is None else bool(use_month)

        self.zone_categories: List[int] = []
        self.time_categories: List[int] = []
        self.conditional_categories: Dict[str, List[int]] = {}

        self.zone_to_idx: Dict[int, int] = {}
        self.time_to_idx: Dict[int, int] = {}
        self.conditional_to_idx: Dict[str, Dict[int, int]] = {}
        self.conditional_columns: List[str] = []

        self._fitted = False

    @property
    def num_zones(self) -> int:
        return len(self.zone_categories)

    @property
    def num_time_bins(self) -> int:
        return len(self.time_categories)

    @property
    def conditional_cardinalities(self) -> Dict[str, int]:
        """Cardinalities keyed by encoded *_idx column names."""
        return {col: len(values) for col, values in self.conditional_categories.items()}

    @property
    def conditional_idx_cardinalities(self) -> Dict[str, int]:
        """Alias for conditional_cardinalities (kept for clarity in training code)."""
        return self.conditional_cardinalities

    def _enabled_conditionals(self) -> List[str]:
        cols = [DOW_COL]
        if self.use_weekend:
            cols.append("is_weekend")
        if self.use_month:
            cols.append("month")
        return cols

    def _conditional_idx_name(self, col: str) -> str:
        if col in CONDITIONAL_IDX_NAMES:
            return CONDITIONAL_IDX_NAMES[col]
        if col in CONDITIONAL_IDX_NAMES.values():
            return col
        raise ValueError(f"Unknown conditional column: {col}")

    def _required_columns(self) -> List[str]:
        return [TIME_COL, PICKUP_COL, DROPOFF_COL, DOW_COL, R_COL] + [
            col for col in ("is_weekend", "month") if col in self._enabled_conditionals()
        ]

    def _ensure_columns(self, df: pd.DataFrame, columns: List[str]) -> None:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def fit(self, train_df: pd.DataFrame) -> "THTTripGenTransformer":
        """Fit categorical mappings using train data only."""
        self._ensure_columns(train_df, self._required_columns())

        zone_values = pd.concat(
            [train_df[PICKUP_COL], train_df[DROPOFF_COL]], ignore_index=True
        )
        zone_values = zone_values.dropna().astype("int64").unique().tolist()
        self.zone_categories = sorted(zone_values)

        time_values = train_df[TIME_COL].dropna().astype("int64").unique().tolist()
        self.time_categories = sorted(time_values)

        conditional_categories: Dict[str, List[int]] = {}
        for col in self._enabled_conditionals():
            values = train_df[col].dropna().astype("int64").unique().tolist()
            idx_name = self._conditional_idx_name(col)
            conditional_categories[idx_name] = sorted(values)

        self.conditional_categories = conditional_categories
        self.conditional_columns = [self._conditional_idx_name(col) for col in self._enabled_conditionals()]

        self.zone_to_idx = {int(val): int(i) for i, val in enumerate(self.zone_categories)}
        self.time_to_idx = {int(val): int(i) for i, val in enumerate(self.time_categories)}
        self.conditional_to_idx = {
            col: {int(val): int(i) for i, val in enumerate(values)}
            for col, values in self.conditional_categories.items()
        }

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, *, drop_unknown: bool = True) -> pd.DataFrame:
        """Transform THT-TripGen dataframe to index space.

        When drop_unknown is True, rows containing any unknown categorical value
        are removed and the output indices are cast to int64.
        """
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before calling transform")

        self._ensure_columns(df, self._required_columns())

        out = pd.DataFrame(index=df.index)
        out["h_idx"] = df[TIME_COL].map(self.time_to_idx)
        out["o_idx"] = df[PICKUP_COL].map(self.zone_to_idx)
        out["d_idx"] = df[DROPOFF_COL].map(self.zone_to_idx)
        out["dow_idx"] = df[DOW_COL].map(self.conditional_to_idx[self._conditional_idx_name(DOW_COL)])

        if self.use_weekend:
            out["is_weekend_idx"] = df["is_weekend"].map(
                self.conditional_to_idx[self._conditional_idx_name("is_weekend")]
            )
        if self.use_month:
            out["month_idx"] = df["month"].map(self.conditional_to_idx[self._conditional_idx_name("month")])

        r = df[R_COL].astype("float32")
        r = r.clip(lower=self.min_r_eps, upper=1.0 - self.min_r_eps)

        if drop_unknown:
            mask = out.notna().all(axis=1) & r.notna()
            out = out[mask].reset_index(drop=True)
            r = r[mask].reset_index(drop=True)

        idx_dtype = "int64" if drop_unknown else "Int64"
        out = out.astype(idx_dtype)
        out[R_COL] = r.astype("float32")
        return out

    def decode(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        """Decode index-space dataframe back to original categorical values."""
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before calling decode")

        required_idx = ["h_idx", "o_idx", "d_idx", "dow_idx", R_COL]
        if self.use_weekend:
            required_idx.append("is_weekend_idx")
        if self.use_month:
            required_idx.append("month_idx")
        self._ensure_columns(df_idx, required_idx)

        out = pd.DataFrame(index=df_idx.index)
        out[TIME_COL] = np.take(
            np.array(self.time_categories, dtype=np.int64),
            df_idx["h_idx"].to_numpy(dtype=np.int64),
        )
        out[PICKUP_COL] = np.take(
            np.array(self.zone_categories, dtype=np.int64),
            df_idx["o_idx"].to_numpy(dtype=np.int64),
        )
        out[DROPOFF_COL] = np.take(
            np.array(self.zone_categories, dtype=np.int64),
            df_idx["d_idx"].to_numpy(dtype=np.int64),
        )
        out[DOW_COL] = np.take(
            np.array(self.conditional_categories[self._conditional_idx_name(DOW_COL)], dtype=np.int64),
            df_idx["dow_idx"].to_numpy(dtype=np.int64),
        )

        if self.use_weekend:
            out["is_weekend"] = np.take(
                np.array(self.conditional_categories[self._conditional_idx_name("is_weekend")], dtype=np.int64),
                df_idx["is_weekend_idx"].to_numpy(dtype=np.int64),
            )
        if self.use_month:
            out["month"] = np.take(
                np.array(self.conditional_categories[self._conditional_idx_name("month")], dtype=np.int64),
                df_idx["month_idx"].to_numpy(dtype=np.int64),
            )

        out[R_COL] = df_idx[R_COL].astype("float32")
        return out

    def state_dict(self) -> THTTripGenTransformerState:
        return THTTripGenTransformerState(
            zone_categories=list(self.zone_categories),
            time_categories=list(self.time_categories),
            conditional_categories={
                key: list(values) for key, values in self.conditional_categories.items()
            },
            conditional_columns=list(self.conditional_columns),
            min_r_eps=float(self.min_r_eps),
            use_weekend=bool(self.use_weekend),
            use_month=bool(self.use_month),
        )

    def load_state_dict(self, state: THTTripGenTransformerState) -> None:
        self.zone_categories = [int(v) for v in state.zone_categories]
        self.time_categories = [int(v) for v in state.time_categories]
        normalized_categories: Dict[str, List[int]] = {}
        for key, values in state.conditional_categories.items():
            idx_name = self._conditional_idx_name(key)
            normalized_categories[idx_name] = [int(v) for v in values]
        self.conditional_categories = normalized_categories
        self.conditional_columns = [
            self._conditional_idx_name(col) for col in state.conditional_columns
        ]
        self.min_r_eps = float(state.min_r_eps)
        self.use_weekend = bool(state.use_weekend)
        self.use_month = bool(state.use_month)

        self.zone_to_idx = {int(val): int(i) for i, val in enumerate(self.zone_categories)}
        self.time_to_idx = {int(val): int(i) for i, val in enumerate(self.time_categories)}
        self.conditional_to_idx = {
            col: {int(val): int(i) for i, val in enumerate(values)}
            for col, values in self.conditional_categories.items()
        }
        self._fitted = True
