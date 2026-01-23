from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from data_processing.strategy_a_transformer import (
    StrategyATransformer,
    StrategyATransformerState,
)
from data_processing.transformer import CategoricalTransformer, TransformerState


def save_mappings(path: str | Path, transformer: CategoricalTransformer) -> None:
    state = transformer.state_dict()
    payload = {
        "columns": state.columns,
        "categories": state.categories,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_mappings(path: str | Path) -> CategoricalTransformer:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    columns = payload["columns"]
    categories = payload["categories"]
    transformer = CategoricalTransformer(columns)
    transformer.load_state_dict(TransformerState(columns=columns, categories=categories))
    return transformer


def save_strategy_a_mappings(
    path: str | Path, transformer: StrategyATransformer
) -> None:
    state = transformer.state_dict()
    payload = {
        "zone_categories": state.zone_categories,
        "time_categories": state.time_categories,
        "conditional_categories": state.conditional_categories,
        "conditional_columns": state.conditional_columns,
        "min_r_eps": state.min_r_eps,
        "use_weekend": state.use_weekend,
        "use_month": state.use_month,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_strategy_a_mappings(path: str | Path) -> StrategyATransformer:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = StrategyATransformerState(
        zone_categories=payload["zone_categories"],
        time_categories=payload["time_categories"],
        conditional_categories=payload["conditional_categories"],
        conditional_columns=payload["conditional_columns"],
        min_r_eps=payload["min_r_eps"],
        use_weekend=payload["use_weekend"],
        use_month=payload["use_month"],
    )
    transformer = StrategyATransformer(
        min_r_eps=state.min_r_eps,
        use_weekend=state.use_weekend,
        use_month=state.use_month,
    )
    transformer.load_state_dict(state)
    return transformer


def save_checkpoint(
    path: str | Path,
    *,
    model_state: Dict[str, Any],
    meta: Dict[str, Any],
    optimizer_state: Dict[str, Any] | None = None,
    epoch: int | None = None,
    metrics: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_state": model_state,
        "meta": meta,
        "optimizer_state": optimizer_state,
        "epoch": epoch,
        "metrics": metrics,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def build_model_from_checkpoint(state: Dict[str, Any]):
    from models.tvae_ar import TVAEAutoregressive

    meta = state["meta"]
    model = TVAEAutoregressive(
        column_sizes=meta["column_sizes"],
        order=meta["order"],
        encoder_hidden_dims=tuple(meta["encoder_hidden_dims"]),
        decoder_hidden_dims=tuple(meta["decoder_hidden_dims"]),
        latent_dim=int(meta["latent_dim"]),
    )
    return model
