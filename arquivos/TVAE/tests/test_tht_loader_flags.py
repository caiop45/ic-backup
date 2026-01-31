import pandas as pd
import pytest


torch = pytest.importorskip("torch")

import config
import train_tht_tripgen


def _make_idx_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "h_idx": [0, 1],
            "o_idx": [0, 1],
            "d_idx": [1, 0],
            "r": [0.1, 0.2],
            "passenger_idx": [0, 1],
            "total_amount_z": [0.0, 0.1],
            "u0": [0, 1],
        }
    )


def test_prepare_loader_cpu_flags(monkeypatch) -> None:
    df_idx = _make_idx_df()
    monkeypatch.setattr(config, "THT_NUM_WORKERS", 2)
    monkeypatch.setattr(config, "THT_PIN_MEMORY", True)
    loader = train_tht_tripgen._prepare_loader(
        df_idx,
        ["u0"],
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
    )
    assert loader.num_workers == 0
    assert loader.pin_memory is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prepare_loader_cuda_flags(monkeypatch) -> None:
    df_idx = _make_idx_df()
    monkeypatch.setattr(config, "THT_NUM_WORKERS", 2)
    monkeypatch.setattr(config, "THT_PIN_MEMORY", True)
    monkeypatch.setattr(config, "THT_PERSISTENT_WORKERS", True)
    monkeypatch.setattr(config, "THT_PREFETCH_FACTOR", 2)
    loader = train_tht_tripgen._prepare_loader(
        df_idx,
        ["u0"],
        device=torch.device("cuda"),
        batch_size=2,
        shuffle=False,
    )
    assert loader.num_workers == 2
    assert loader.pin_memory is True
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == config.THT_PREFETCH_FACTOR
