from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class TransformerState:
    columns: List[str]
    categories: Dict[str, List[int]]


class CategoricalTransformer:
    def __init__(self, columns: Iterable[str]):
        self.columns = list(columns)
        self.categories: Dict[str, List[int]] = {}
        self.category_to_index: Dict[str, Dict[int, int]] = {}
        self.cardinalities: Dict[str, int] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CategoricalTransformer":
        categories: Dict[str, List[int]] = {}
        category_to_index: Dict[str, Dict[int, int]] = {}
        cardinalities: Dict[str, int] = {}

        for col in self.columns:
            values = df[col].dropna().astype("int64").unique()
            values_sorted = sorted(values.tolist())
            categories[col] = values_sorted
            category_to_index[col] = {v: i for i, v in enumerate(values_sorted)}
            cardinalities[col] = len(values_sorted)

        self.categories = categories
        self.category_to_index = category_to_index
        self.cardinalities = cardinalities
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, *, drop_unknown: bool = True) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before calling transform")

        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            mapping = self.category_to_index[col]
            idx = df[col].map(mapping)
            out[col] = idx

        if drop_unknown:
            before = len(out)
            out = out.dropna().reset_index(drop=True)
            removed = before - len(out)
            if removed > 0:
                print(f"[Transformer] Dropped {removed} rows with unknown categories")

        out = out.astype("int64")
        return out

    def one_hot_encode(self, df_idx: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before calling one_hot_encode")

        n = len(df_idx)
        parts: List[np.ndarray] = []
        for col in self.columns:
            idx = df_idx[col].to_numpy(dtype=np.int64)
            k = self.cardinalities[col]
            one_hot = np.zeros((n, k), dtype=np.float32)
            if n > 0:
                one_hot[np.arange(n), idx] = 1.0
            parts.append(one_hot)

        if not parts:
            return np.zeros((n, 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    def decode_indices(self, df_idx: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before calling decode_indices")

        out = pd.DataFrame(index=df_idx.index)
        for col in self.columns:
            categories = self.categories[col]
            idx = df_idx[col].to_numpy(dtype=np.int64)
            out[col] = np.take(np.array(categories, dtype=np.int64), idx)
        return out

    def decode_array(self, indices: np.ndarray, *, column: str) -> np.ndarray:
        if column not in self.categories:
            raise KeyError(f"Unknown column: {column}")
        categories = np.array(self.categories[column], dtype=np.int64)
        return categories[indices]

    def state_dict(self) -> TransformerState:
        return TransformerState(columns=self.columns, categories=self.categories)

    def load_state_dict(self, state: TransformerState) -> None:
        self.columns = list(state.columns)
        self.categories = {k: list(v) for k, v in state.categories.items()}
        self.category_to_index = {
            k: {int(val): int(i) for i, val in enumerate(v)}
            for k, v in self.categories.items()
        }
        self.cardinalities = {k: len(v) for k, v in self.categories.items()}
        self._fitted = True
