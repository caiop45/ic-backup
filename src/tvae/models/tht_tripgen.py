"""THT-TripGen discrete generator: p(h|u) p(o|h,u) p(d|o,h,u).

Destination head options:
- embedding_softmax: projection into frozen zone embedding space.
- linear: free linear logits over zones.
- hybrid: embedding_softmax logits plus a learned residual logits head.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tvae import config
from tvae.models.flows.conditional_spline_1d import ConditionalSplineFlow1D


@dataclass(frozen=True)
class THTTripGenInputs:
    u: Dict[str, torch.Tensor]
    h_idx: torch.Tensor
    o_idx: torch.Tensor
    d_idx: torch.Tensor


def _ensure_1d_long(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    return tensor.to(dtype=torch.long)


def _make_mlp(
    input_dim: int, hidden_dim: int, layers: int, dropout: float
) -> nn.Sequential:
    if layers <= 0:
        raise ValueError("layers must be positive")
    modules: List[nn.Module] = []
    dim = input_dim
    for _ in range(layers):
        modules.append(nn.Linear(dim, hidden_dim))
        modules.append(nn.GELU())
        if dropout > 0:
            modules.append(nn.Dropout(dropout))
        dim = hidden_dim
    return nn.Sequential(*modules)


class EmbeddingSoftmaxHead(nn.Module):
    """Embedding-softmax logits using frozen zone embeddings."""

    def __init__(self, ctx_dim: int, *, emb_dim: int, num_zones: int) -> None:
        super().__init__()
        self.proj = nn.Linear(ctx_dim, emb_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_zones))

    def forward(self, ctx: torch.Tensor, zone_embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(ctx) @ zone_embeddings.t() + self.bias


class LinearDestinationHead(nn.Module):
    """Unconstrained linear logits over zones."""

    def __init__(self, ctx_dim: int, *, num_zones: int) -> None:
        super().__init__()
        self.linear = nn.Linear(ctx_dim, num_zones)

    def forward(self, ctx: torch.Tensor, zone_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        _ = zone_embeddings
        return self.linear(ctx)


class HybridDestinationHead(nn.Module):
    """Embedding-softmax logits plus a learned residual logits head."""

    def __init__(
        self,
        ctx_dim: int,
        *,
        emb_dim: int,
        num_zones: int,
        residual_hidden: int,
        dropout: float,
        residual_weight_init: float,
    ) -> None:
        super().__init__()
        self.embed_head = EmbeddingSoftmaxHead(ctx_dim, emb_dim=emb_dim, num_zones=num_zones)
        if residual_hidden > 0:
            self.residual_mlp = _make_mlp(ctx_dim, residual_hidden, layers=1, dropout=dropout)
            self.residual_out = nn.Linear(residual_hidden, num_zones)
        else:
            self.residual_mlp = nn.Identity()
            self.residual_out = nn.Linear(ctx_dim, num_zones)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_weight_init)))

    def forward(self, ctx: torch.Tensor, zone_embeddings: torch.Tensor) -> torch.Tensor:
        base = self.embed_head(ctx, zone_embeddings)
        residual = self.residual_out(self.residual_mlp(ctx))
        return base + self.residual_scale * residual


def _normalize_destination_head_type(value: str) -> str:
    name = value.strip().lower()
    if name in ("free_logits", "linear"):
        return "linear"
    if name in ("embedding_softmax", "hybrid"):
        return name
    raise ValueError("destination_head_type must be 'embedding_softmax', 'linear', or 'hybrid'")


class THTTripGenModel(nn.Module):
    """THT-TripGen model implementing time-first factorization.

    p(h|u) p(o|h,u) p(d|o,h,u)

    Destination heads:
    - embedding_softmax
    - linear
    - hybrid

    Optional attribute heads:
    - passenger_count (categorical)
    - total_amount (continuous, standardized)
    """

    def __init__(
        self,
        *,
        num_zones: int,
        num_time_bins: int,
        conditional_cardinalities: Dict[str, int],
        frozen_zone_embeddings: torch.Tensor,
        cond_emb_dim: int = config.THT_COND_EMB_DIM,
        time_emb_dim: int = config.THT_TIME_EMB_DIM,
        origin_emb_dim: int = config.THT_ORIGIN_EMB_DIM,
        context_mlp_hidden: int = config.THT_MODEL_HIDDEN,
        context_mlp_layers: int = config.THT_MODEL_LAYERS,
        dropout: float = config.THT_MODEL_DROPOUT,
        destination_head_type: str = config.THT_DEST_HEAD_TYPE,
        hybrid_residual_hidden: int = config.THT_HYBRID_RESIDUAL_HIDDEN,
        hybrid_residual_weight_init: float = config.THT_HYBRID_RESIDUAL_WEIGHT_INIT,
        use_passenger_count: bool = config.THT_USE_PASSENGER_COUNT,
        passenger_cardinality: int | None = None,
        use_total_amount: bool = config.THT_USE_TOTAL_AMOUNT,
        total_amount_sigma_floor: float = config.THT_TOTAL_AMOUNT_SIGMA_FLOOR,
        min_r_eps: float = config.THT_MIN_R_EPS,
        residual_num_layers: int = config.THT_RESIDUAL_NUM_LAYERS,
        residual_num_bins: int = config.THT_RESIDUAL_NUM_BINS,
        residual_context_hidden: int = config.THT_RESIDUAL_CONTEXT_HIDDEN,
        residual_min_bin_width: float = config.THT_RESIDUAL_MIN_BIN_WIDTH,
        residual_min_bin_height: float = config.THT_RESIDUAL_MIN_BIN_HEIGHT,
        residual_min_deriv: float = config.THT_RESIDUAL_MIN_DERIV,
        residual_eps: float = config.THT_RESIDUAL_EPS,
    ) -> None:
        super().__init__()

        if num_zones <= 0:
            raise ValueError("num_zones must be positive")
        if num_time_bins <= 0:
            raise ValueError("num_time_bins must be positive")
        if frozen_zone_embeddings.dim() != 2:
            raise ValueError("frozen_zone_embeddings must be a 2D tensor")
        if int(frozen_zone_embeddings.shape[0]) != int(num_zones):
            raise ValueError("frozen_zone_embeddings row count must match num_zones")
        destination_head_type = _normalize_destination_head_type(destination_head_type)

        self.num_zones = int(num_zones)
        self.num_time_bins = int(num_time_bins)
        self.conditional_cardinalities = {
            str(k): int(v) for k, v in conditional_cardinalities.items()
        }
        self.destination_head_type = destination_head_type
        self.min_r_eps = float(min_r_eps)
        self.use_passenger_count = bool(use_passenger_count)
        self.use_total_amount = bool(use_total_amount)
        self.total_amount_sigma_floor = float(total_amount_sigma_floor)

        self.conditional_names = list(self.conditional_cardinalities.keys())
        self.conditional_idx_names = [
            name if name.endswith("_idx") else f"{name}_idx" for name in self.conditional_names
        ]
        self._cond_idx_map = {
            name: idx_name
            for name, idx_name in zip(self.conditional_names, self.conditional_idx_names)
        }

        self.cond_embeddings = nn.ModuleDict()
        for name, idx_name in zip(self.conditional_names, self.conditional_idx_names):
            card = self.conditional_cardinalities[name]
            if card <= 0:
                raise ValueError(f"conditional_cardinalities[{name}] must be positive")
            self.cond_embeddings[idx_name] = nn.Embedding(card, cond_emb_dim)

        self.time_embedding = nn.Embedding(self.num_time_bins, time_emb_dim)
        self.origin_embedding = nn.Embedding(self.num_zones, origin_emb_dim)

        u_dim = cond_emb_dim * len(self.conditional_idx_names)
        ho_input_dim = u_dim + time_emb_dim
        d_input_dim = u_dim + time_emb_dim + origin_emb_dim

        self.mlp_ho = _make_mlp(ho_input_dim, context_mlp_hidden, context_mlp_layers, dropout)
        self.mlp_d = _make_mlp(d_input_dim, context_mlp_hidden, context_mlp_layers, dropout)

        self.head_h = nn.Linear(u_dim, self.num_time_bins)
        self.head_o = nn.Linear(context_mlp_hidden, self.num_zones)

        emb_dim = int(frozen_zone_embeddings.shape[1])
        self.register_buffer(
            "zone_embeddings",
            frozen_zone_embeddings.detach().clone().to(dtype=torch.float32),
            persistent=True,
        )
        if destination_head_type == "embedding_softmax":
            self.dest_project = nn.Linear(context_mlp_hidden, emb_dim, bias=False)
            self.dest_bias = nn.Parameter(torch.zeros(self.num_zones))
            self.dest_free = None
            self.dest_hybrid = None
        elif destination_head_type == "hybrid":
            self.dest_project = None
            self.dest_bias = None
            self.dest_free = None
            self.dest_hybrid = HybridDestinationHead(
                context_mlp_hidden,
                emb_dim=emb_dim,
                num_zones=self.num_zones,
                residual_hidden=int(hybrid_residual_hidden),
                dropout=float(dropout),
                residual_weight_init=float(hybrid_residual_weight_init),
            )
        else:
            self.dest_project = None
            self.dest_bias = None
            self.dest_free = nn.Linear(context_mlp_hidden, self.num_zones)
            self.dest_hybrid = None

        residual_input_dim = u_dim + time_emb_dim + origin_emb_dim + emb_dim
        self.residual_mlp = _make_mlp(
            residual_input_dim, residual_context_hidden, context_mlp_layers, dropout
        )
        self.residual_flow = ConditionalSplineFlow1D(
            context_dim=residual_context_hidden,
            num_layers=residual_num_layers,
            num_bins=residual_num_bins,
            min_bin_width=residual_min_bin_width,
            min_bin_height=residual_min_bin_height,
            min_derivative=residual_min_deriv,
            eps=residual_eps,
        )

        if self.use_passenger_count:
            if passenger_cardinality is None or int(passenger_cardinality) <= 0:
                raise ValueError("passenger_cardinality must be positive when enabled")
            self.passenger_head = nn.Linear(
                residual_context_hidden, int(passenger_cardinality)
            )
        else:
            self.passenger_head = None

        if self.use_total_amount:
            self.amount_head = nn.Linear(residual_context_hidden, 2)
        else:
            self.amount_head = None

    def _normalize_u(self, u: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        normalized: Dict[str, torch.Tensor] = {}
        for name, idx_name in self._cond_idx_map.items():
            if idx_name in u:
                normalized[idx_name] = u[idx_name]
            elif name in u:
                normalized[idx_name] = u[name]
            else:
                raise ValueError(f"Missing conditional input: {idx_name}")
        return normalized

    def _u_embed(self, u: Dict[str, torch.Tensor]) -> torch.Tensor:
        normalized = self._normalize_u(u)
        parts: List[torch.Tensor] = []
        expected_len: int | None = None
        for idx_name in self.conditional_idx_names:
            idx = _ensure_1d_long(normalized[idx_name], idx_name)
            if expected_len is None:
                expected_len = idx.numel()
            elif idx.numel() != expected_len:
                raise ValueError("All conditional inputs must share the same length")
            parts.append(self.cond_embeddings[idx_name](idx))
        if not parts:
            batch = expected_len or 0
            return torch.empty((batch, 0), device=next(self.parameters()).device)
        return torch.cat(parts, dim=1)

    def forward(self, *, u: Dict[str, torch.Tensor], h_idx: torch.Tensor, o_idx: torch.Tensor, d_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        u_emb = self._u_embed(u)

        h_idx = _ensure_1d_long(h_idx, "h_idx")
        o_idx = _ensure_1d_long(o_idx, "o_idx")
        d_idx = _ensure_1d_long(d_idx, "d_idx")
        if u_emb.shape[0] != h_idx.numel() or u_emb.shape[0] != o_idx.numel() or u_emb.shape[0] != d_idx.numel():
            raise ValueError("u, h_idx, o_idx, and d_idx must share the same batch size")

        h_emb = self.time_embedding(h_idx)

        logits_h = self.head_h(u_emb)
        ho_ctx = self.mlp_ho(torch.cat([u_emb, h_emb], dim=1))
        logits_o = self.head_o(ho_ctx)

        o_emb = self.origin_embedding(o_idx)
        d_ctx = self.mlp_d(torch.cat([u_emb, h_emb, o_emb], dim=1))

        if self.destination_head_type == "embedding_softmax":
            proj = self.dest_project(d_ctx)
            logits_d = proj @ self.zone_embeddings.t() + self.dest_bias
        elif self.destination_head_type == "hybrid":
            if self.dest_hybrid is None:
                raise RuntimeError("Hybrid destination head not initialized")
            logits_d = self.dest_hybrid(d_ctx, self.zone_embeddings)
        else:
            logits_d = self.dest_free(d_ctx)

        return {
            "logits_h": logits_h,
            "logits_o": logits_o,
            "logits_d": logits_d,
        }

    def nll(
        self,
        *,
        u: Dict[str, torch.Tensor],
        h_idx: torch.Tensor,
        o_idx: torch.Tensor,
        d_idx: torch.Tensor,
        r: torch.Tensor,
        passenger_idx: torch.Tensor | None = None,
        total_amount_z: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.forward(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx)

        nll_h = F.cross_entropy(logits["logits_h"], _ensure_1d_long(h_idx, "h_idx"))
        nll_o = F.cross_entropy(logits["logits_o"], _ensure_1d_long(o_idx, "o_idx"))
        nll_d = F.cross_entropy(logits["logits_d"], _ensure_1d_long(d_idx, "d_idx"))

        u_emb = self._u_embed(u)
        h_emb = self.time_embedding(_ensure_1d_long(h_idx, "h_idx"))
        o_emb = self.origin_embedding(_ensure_1d_long(o_idx, "o_idx"))
        d_emb = self.zone_embeddings[_ensure_1d_long(d_idx, "d_idx")]
        residual_ctx = self.residual_mlp(torch.cat([u_emb, h_emb, o_emb, d_emb], dim=1))

        r = r.to(dtype=torch.float32)
        log_prob_r = self.residual_flow.log_prob(r, residual_ctx)
        nll_r = -log_prob_r.mean()

        if self.use_passenger_count:
            if passenger_idx is None:
                raise ValueError("passenger_idx is required when passenger_count is enabled")
            logits_pass = self.passenger_head(residual_ctx)
            nll_passenger = F.cross_entropy(
                logits_pass, _ensure_1d_long(passenger_idx, "passenger_idx")
            )
        else:
            nll_passenger = torch.zeros((), device=nll_h.device)

        if self.use_total_amount:
            if total_amount_z is None:
                raise ValueError("total_amount_z is required when total_amount is enabled")
            params = self.amount_head(residual_ctx)
            mu = params[:, 0]
            log_sigma = params[:, 1]
            sigma = F.softplus(log_sigma) + self.total_amount_sigma_floor
            z = total_amount_z.to(dtype=torch.float32)
            log_term = torch.log(sigma)
            quad = 0.5 * ((z - mu) / sigma).pow(2)
            nll_total_amount = (0.5 * math.log(2.0 * math.pi) + log_term + quad).mean()
        else:
            nll_total_amount = torch.zeros((), device=nll_h.device)

        total = nll_h + nll_o + nll_d + nll_r + nll_passenger + nll_total_amount
        return {
            "nll_h": nll_h,
            "nll_o": nll_o,
            "nll_d": nll_d,
            "nll_r": nll_r,
            "nll_passenger": nll_passenger,
            "nll_total_amount": nll_total_amount,
            "nll_total": total,
        }

    def residual_log_prob(
        self,
        *,
        u: Dict[str, torch.Tensor],
        h_idx: torch.Tensor,
        o_idx: torch.Tensor,
        d_idx: torch.Tensor,
        r: torch.Tensor,
        return_layer_logdet: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        """Compute residual flow log-probability with optional layer diagnostics."""
        u_emb = self._u_embed(u)

        h_idx = _ensure_1d_long(h_idx, "h_idx")
        o_idx = _ensure_1d_long(o_idx, "o_idx")
        d_idx = _ensure_1d_long(d_idx, "d_idx")
        if (
            u_emb.shape[0] != h_idx.numel()
            or u_emb.shape[0] != o_idx.numel()
            or u_emb.shape[0] != d_idx.numel()
        ):
            raise ValueError("u, h_idx, o_idx, and d_idx must share the same batch size")

        h_emb = self.time_embedding(h_idx)
        o_emb = self.origin_embedding(o_idx)
        d_emb = self.zone_embeddings[d_idx]
        residual_ctx = self.residual_mlp(torch.cat([u_emb, h_emb, o_emb, d_emb], dim=1))

        r = r.to(dtype=torch.float32)
        return self.residual_flow.log_prob(
            r, residual_ctx, return_layer_logdet=return_layer_logdet
        )

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        scaled = logits / float(temperature)
        probs = F.softmax(scaled, dim=1)
        idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=generator)
        return idx.squeeze(1)

    def _prepare_fixed(
        self, value: torch.Tensor | int | None, n: int, *, name: str, device: torch.device
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, int):
            return torch.full((n,), value, dtype=torch.long, device=device)
        tensor = value.to(device=device, dtype=torch.long)
        if tensor.dim() != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        if tensor.numel() == 1 and n > 1:
            return tensor.repeat(n)
        if tensor.numel() != n:
            raise ValueError(f"{name} length must be {n} or 1")
        return tensor

    def sample(
        self,
        n: int,
        *,
        u: Dict[str, torch.Tensor | int] | None = None,
        h: torch.Tensor | int | None = None,
        o: torch.Tensor | int | None = None,
        temperature: float = config.THT_SAMPLE_TEMPERATURE,
        seed: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive")

        device = next(self.parameters()).device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        training = self.training
        self.eval()
        with torch.no_grad():
            # Prepare u
            if u is None:
                u = {}
                for name, idx_name in self._cond_idx_map.items():
                    card = self.conditional_cardinalities[name]
                    u[idx_name] = torch.randint(
                        0, card, (n,), device=device, generator=generator
                    )
            else:
                normalized: Dict[str, torch.Tensor] = {}
                for name, idx_name in self._cond_idx_map.items():
                    if idx_name in u:
                        val = u[idx_name]
                    elif name in u:
                        val = u[name]
                    else:
                        raise ValueError(f"Missing conditional input: {idx_name}")

                    if isinstance(val, int):
                        normalized[idx_name] = torch.full(
                            (n,), val, dtype=torch.long, device=device
                        )
                    else:
                        tensor = val.to(device=device, dtype=torch.long)
                        if tensor.dim() != 1:
                            raise ValueError(f"{idx_name} must be a 1D tensor")
                        if tensor.numel() == 1 and n > 1:
                            tensor = tensor.repeat(n)
                        if tensor.numel() != n:
                            raise ValueError(f"{idx_name} length must be {n} or 1")
                        normalized[idx_name] = tensor
                u = normalized

            u_emb = self._u_embed(u)

            # Sample h
            h_tensor = self._prepare_fixed(h, n, name="h", device=device)
            if h_tensor is None:
                logits_h = self.head_h(u_emb)
                h_tensor = self._sample_from_logits(
                    logits_h, temperature, generator=generator
                )

            # Sample o
            o_tensor = self._prepare_fixed(o, n, name="o", device=device)
            if o_tensor is None:
                h_emb = self.time_embedding(h_tensor)
                ho_ctx = self.mlp_ho(torch.cat([u_emb, h_emb], dim=1))
                logits_o = self.head_o(ho_ctx)
                o_tensor = self._sample_from_logits(
                    logits_o, temperature, generator=generator
                )

            # Sample d
            h_emb = self.time_embedding(h_tensor)
            o_emb = self.origin_embedding(o_tensor)
            d_ctx = self.mlp_d(torch.cat([u_emb, h_emb, o_emb], dim=1))
            if self.destination_head_type == "embedding_softmax":
                proj = self.dest_project(d_ctx)
                logits_d = proj @ self.zone_embeddings.t() + self.dest_bias
            elif self.destination_head_type == "hybrid":
                if self.dest_hybrid is None:
                    raise RuntimeError("Hybrid destination head not initialized")
                logits_d = self.dest_hybrid(d_ctx, self.zone_embeddings)
            else:
                logits_d = self.dest_free(d_ctx)
            d_tensor = self._sample_from_logits(
                logits_d, temperature, generator=generator
            )

            d_emb = self.zone_embeddings[d_tensor]
            residual_ctx = self.residual_mlp(torch.cat([u_emb, h_emb, o_emb, d_emb], dim=1))
            r = self.residual_flow.sample(residual_ctx, seed=seed)

            output: Dict[str, torch.Tensor] = {
                "h_idx": h_tensor,
                "o_idx": o_tensor,
                "d_idx": d_tensor,
                "r": r,
            }

            if self.use_passenger_count:
                if self.passenger_head is None:
                    raise RuntimeError("Passenger head not initialized")
                logits_pass = self.passenger_head(residual_ctx)
                passenger_idx = self._sample_from_logits(
                    logits_pass, temperature, generator=generator
                )
                output["passenger_idx"] = passenger_idx

            if self.use_total_amount:
                if self.amount_head is None:
                    raise RuntimeError("Amount head not initialized")
                params = self.amount_head(residual_ctx)
                mu = params[:, 0]
                log_sigma = params[:, 1]
                sigma = F.softplus(log_sigma) + self.total_amount_sigma_floor
                eps = torch.randn(mu.shape, device=mu.device, generator=generator)
                total_amount_z = mu + sigma * eps
                output["total_amount_z"] = total_amount_z

            for idx_name in self.conditional_idx_names:
                output[idx_name] = u[idx_name]

        if training:
            self.train()

        return output
