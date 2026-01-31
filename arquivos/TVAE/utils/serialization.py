from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict, NotRequired

import torch

from data_processing.tht_tripgen_transformer import (
    THTTripGenTransformer,
    THTTripGenTransformerState,
)
from data_processing.transformer import CategoricalTransformer, TransformerState
from models.tht_tripgen import THTTripGenModel
import config


# THT-TripGen artifact filenames for run directories.
THT_TRIPGEN_CHECKPOINT_FILENAME = "tht_tripgen.pt"
THT_TRIPGEN_MAPPINGS_FILENAME = "mappings_tht_tripgen.json"
THT_TRIPGEN_METRICS_FILENAME = "metrics_tht_tripgen.json"
THT_TRIPGEN_METRICS_HOLD_FILENAME = "metrics_tht_tripgen_hold.json"
THT_TRIPGEN_LOSS_FILENAME = "loss_tht_tripgen.csv"
THT_TRIPGEN_SYNTHETIC_HOLD_FILENAME = "synthetic_tht_tripgen_hold.csv"


class THTTripGenCheckpointMeta(TypedDict):
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
    hybrid_residual_hidden: NotRequired[int]
    hybrid_residual_weight_init: NotRequired[float]
    use_passenger_count: NotRequired[bool]
    passenger_cardinality: NotRequired[int]
    use_total_amount: NotRequired[bool]
    total_amount_sigma_floor: NotRequired[float]
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


def save_tht_tripgen_mappings(
    path: str | Path, transformer: THTTripGenTransformer
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
        "use_passenger_count": state.use_passenger_count,
        "use_total_amount": state.use_total_amount,
        "passenger_categories": state.passenger_categories,
        "total_amount_mu": state.total_amount_mu,
        "total_amount_sigma": state.total_amount_sigma,
        "total_amount_clip_lo": state.total_amount_clip_lo,
        "total_amount_clip_hi": state.total_amount_clip_hi,
        "total_amount_log1p": state.total_amount_log1p,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def load_tht_tripgen_mappings(path: str | Path) -> THTTripGenTransformer:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    state = THTTripGenTransformerState(
        zone_categories=payload["zone_categories"],
        time_categories=payload["time_categories"],
        conditional_categories=payload["conditional_categories"],
        conditional_columns=payload["conditional_columns"],
        min_r_eps=payload["min_r_eps"],
        use_weekend=payload["use_weekend"],
        use_month=payload["use_month"],
        use_passenger_count=payload.get("use_passenger_count", False),
        use_total_amount=payload.get("use_total_amount", False),
        passenger_categories=payload.get("passenger_categories", []),
        total_amount_mu=payload.get("total_amount_mu"),
        total_amount_sigma=payload.get("total_amount_sigma"),
        total_amount_clip_lo=payload.get("total_amount_clip_lo"),
        total_amount_clip_hi=payload.get("total_amount_clip_hi"),
        total_amount_log1p=payload.get("total_amount_log1p", True),
    )
    transformer = THTTripGenTransformer(
        min_r_eps=state.min_r_eps,
        use_weekend=state.use_weekend,
        use_month=state.use_month,
        use_passenger_count=state.use_passenger_count,
        use_total_amount=state.use_total_amount,
        total_amount_log1p=state.total_amount_log1p,
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


def build_tht_tripgen_model_from_checkpoint(state: Dict[str, Any]) -> THTTripGenModel:
    meta = state.get("meta")
    if meta is None:
        raise ValueError("Checkpoint state missing meta for THT-TripGen")

    conditional_cardinalities = meta.get("conditional_cardinalities")
    if conditional_cardinalities is None:
        raise ValueError("meta.conditional_cardinalities is required for THT-TripGen")

    num_zones = meta.get("num_zones")
    num_time_bins = meta.get("num_time_bins")
    if num_zones is None or num_time_bins is None:
        raise ValueError("meta.num_zones and meta.num_time_bins are required for THT-TripGen")

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

    use_passenger_count = bool(meta.get("use_passenger_count", False))
    use_total_amount = bool(meta.get("use_total_amount", False))

    model = THTTripGenModel(
        num_zones=int(num_zones),
        num_time_bins=int(num_time_bins),
        conditional_cardinalities={
            str(k): int(v) for k, v in conditional_cardinalities.items()
        },
        frozen_zone_embeddings=zone_embeddings,
        cond_emb_dim=int(_meta_or_default("cond_emb_dim", config.THT_COND_EMB_DIM)),
        time_emb_dim=int(_meta_or_default("time_emb_dim", config.THT_TIME_EMB_DIM)),
        origin_emb_dim=int(_meta_or_default("origin_emb_dim", config.THT_ORIGIN_EMB_DIM)),
        context_mlp_hidden=int(_meta_or_default("context_mlp_hidden", config.THT_MODEL_HIDDEN)),
        context_mlp_layers=int(_meta_or_default("context_mlp_layers", config.THT_MODEL_LAYERS)),
        dropout=float(_meta_or_default("dropout", config.THT_MODEL_DROPOUT)),
        destination_head_type=str(
            _meta_or_default("destination_head_type", config.THT_DEST_HEAD_TYPE)
        ),
        hybrid_residual_hidden=int(
            _meta_or_default("hybrid_residual_hidden", config.THT_HYBRID_RESIDUAL_HIDDEN)
        ),
        hybrid_residual_weight_init=float(
            _meta_or_default(
                "hybrid_residual_weight_init", config.THT_HYBRID_RESIDUAL_WEIGHT_INIT
            )
        ),
        use_passenger_count=use_passenger_count,
        passenger_cardinality=(
            int(_meta_or_default("passenger_cardinality", 0)) if use_passenger_count else None
        ),
        use_total_amount=use_total_amount,
        total_amount_sigma_floor=float(
            _meta_or_default(
                "total_amount_sigma_floor", config.THT_TOTAL_AMOUNT_SIGMA_FLOOR
            )
        ),
        min_r_eps=float(_meta_or_default("min_r_eps", config.THT_MIN_R_EPS)),
        residual_num_layers=int(
            _meta_or_default("residual_num_layers", config.THT_RESIDUAL_NUM_LAYERS)
        ),
        residual_num_bins=int(
            _meta_or_default("residual_num_bins", config.THT_RESIDUAL_NUM_BINS)
        ),
        residual_context_hidden=int(
            _meta_or_default("residual_context_hidden", config.THT_RESIDUAL_CONTEXT_HIDDEN)
        ),
        residual_min_bin_width=float(
            _meta_or_default("residual_min_bin_width", config.THT_RESIDUAL_MIN_BIN_WIDTH)
        ),
        residual_min_bin_height=float(
            _meta_or_default("residual_min_bin_height", config.THT_RESIDUAL_MIN_BIN_HEIGHT)
        ),
        residual_min_deriv=float(
            _meta_or_default("residual_min_deriv", config.THT_RESIDUAL_MIN_DERIV)
        ),
        residual_eps=float(_meta_or_default("residual_eps", config.THT_RESIDUAL_EPS)),
    )
    return model
