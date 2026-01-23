from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict, NotRequired

import torch

from data_processing.strategy_a_transformer import (
    StrategyATransformer,
    StrategyATransformerState,
)
from data_processing.transformer import CategoricalTransformer, TransformerState
from models.strategy_a import StrategyAModel
import config


# Strategy A artifact filenames for run directories.
STRATEGY_A_CHECKPOINT_FILENAME = "strategy_a.pt"
STRATEGY_A_MAPPINGS_FILENAME = "mappings_strategy_a.json"
STRATEGY_A_METRICS_FILENAME = "metrics_strategy_a.json"
STRATEGY_A_METRICS_HOLD_FILENAME = "metrics_strategy_a_hold.json"
STRATEGY_A_LOSS_FILENAME = "loss_strategy_a.csv"
STRATEGY_A_SYNTHETIC_HOLD_FILENAME = "synthetic_strategy_a_hold.csv"


class StrategyACheckpointMeta(TypedDict):
    num_zones: int
    num_time_bins: int
    conditional_cardinalities: Dict[str, int]
    zone_embeddings_source: str
    zone_embeddings_dim: int
    zone_embeddings_path: NotRequired[str]
    cond_emb_dim: NotRequired[int]
    time_emb_dim: NotRequired[int]
    origin_emb_dim: NotRequired[int]
    context_mlp_hidden: NotRequired[int]
    context_mlp_layers: NotRequired[int]
    dropout: NotRequired[float]
    destination_head_type: NotRequired[str]
    min_r_eps: NotRequired[float]
    residual_num_layers: NotRequired[int]
    residual_num_bins: NotRequired[int]
    residual_context_hidden: NotRequired[int]
    residual_min_bin_width: NotRequired[float]
    residual_min_bin_height: NotRequired[float]
    residual_min_deriv: NotRequired[float]
    residual_eps: NotRequired[float]


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


def build_strategy_a_model_from_checkpoint(state: Dict[str, Any]) -> StrategyAModel:
    meta = state.get("meta")
    if meta is None:
        raise ValueError("Checkpoint state missing meta for Strategy A")

    conditional_cardinalities = meta.get("conditional_cardinalities")
    if conditional_cardinalities is None:
        raise ValueError("meta.conditional_cardinalities is required for Strategy A")

    num_zones = meta.get("num_zones")
    num_time_bins = meta.get("num_time_bins")
    if num_zones is None or num_time_bins is None:
        raise ValueError("meta.num_zones and meta.num_time_bins are required for Strategy A")

    model_state = state.get("model_state", {})
    zone_embeddings = None
    if isinstance(model_state, dict):
        zone_embeddings = model_state.get("zone_embeddings")

    if zone_embeddings is None:
        emb_path = meta.get("zone_embeddings_path") or meta.get("frozen_zone_embeddings_path")
        if emb_path:
            zone_embeddings = torch.load(emb_path, map_location="cpu")
        else:
            raw_embeddings = meta.get("zone_embeddings")
            if raw_embeddings is not None:
                zone_embeddings = torch.as_tensor(raw_embeddings, dtype=torch.float32)

    if zone_embeddings is None:
        raise ValueError(
            "Zone embeddings missing from checkpoint: expected model_state['zone_embeddings'] "
            "or meta zone_embeddings_path/zone_embeddings."
        )

    if not isinstance(zone_embeddings, torch.Tensor):
        zone_embeddings = torch.as_tensor(zone_embeddings, dtype=torch.float32)

    def _meta_or_default(key: str, default: Any) -> Any:
        value = meta.get(key)
        return default if value is None else value

    model = StrategyAModel(
        num_zones=int(num_zones),
        num_time_bins=int(num_time_bins),
        conditional_cardinalities={
            str(k): int(v) for k, v in conditional_cardinalities.items()
        },
        frozen_zone_embeddings=zone_embeddings,
        cond_emb_dim=int(_meta_or_default("cond_emb_dim", config.SA_COND_EMB_DIM)),
        time_emb_dim=int(_meta_or_default("time_emb_dim", config.SA_TIME_EMB_DIM)),
        origin_emb_dim=int(_meta_or_default("origin_emb_dim", config.SA_ORIGIN_EMB_DIM)),
        context_mlp_hidden=int(_meta_or_default("context_mlp_hidden", config.SA_MODEL_HIDDEN)),
        context_mlp_layers=int(_meta_or_default("context_mlp_layers", config.SA_MODEL_LAYERS)),
        dropout=float(_meta_or_default("dropout", config.SA_MODEL_DROPOUT)),
        destination_head_type=str(
            _meta_or_default("destination_head_type", config.SA_DEST_HEAD_TYPE)
        ),
        min_r_eps=float(_meta_or_default("min_r_eps", config.SA_MIN_R_EPS)),
        residual_num_layers=int(
            _meta_or_default("residual_num_layers", config.SA_RESIDUAL_NUM_LAYERS)
        ),
        residual_num_bins=int(
            _meta_or_default("residual_num_bins", config.SA_RESIDUAL_NUM_BINS)
        ),
        residual_context_hidden=int(
            _meta_or_default("residual_context_hidden", config.SA_RESIDUAL_CONTEXT_HIDDEN)
        ),
        residual_min_bin_width=float(
            _meta_or_default("residual_min_bin_width", config.SA_RESIDUAL_MIN_BIN_WIDTH)
        ),
        residual_min_bin_height=float(
            _meta_or_default("residual_min_bin_height", config.SA_RESIDUAL_MIN_BIN_HEIGHT)
        ),
        residual_min_deriv=float(
            _meta_or_default("residual_min_deriv", config.SA_RESIDUAL_MIN_DERIV)
        ),
        residual_eps=float(_meta_or_default("residual_eps", config.SA_RESIDUAL_EPS)),
    )
    return model
