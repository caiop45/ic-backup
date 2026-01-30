from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch


def set_seed(
    seed: int,
    *,
    cudnn_deterministic: bool | None = None,
    cudnn_benchmark: bool | None = None,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic is None or cudnn_benchmark is None:
        try:
            import config as _config

            det = bool(getattr(_config, "CUDNN_DETERMINISTIC", False))
            bench = bool(getattr(_config, "CUDNN_BENCHMARK", True))
        except Exception:
            det = False
            bench = True
    else:
        det = bool(cudnn_deterministic)
        bench = bool(cudnn_benchmark)

    # Deterministic mode overrides benchmark for reproducibility.
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = False if det else bench


def safe_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, 1e-12, None)


def concat_arrays(arrays: Iterable[np.ndarray], axis: int = 1) -> np.ndarray:
    arrays = list(arrays)
    if not arrays:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(arrays, axis=axis)
