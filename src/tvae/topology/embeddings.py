"""Utilities for normalizing and combining embeddings."""
from __future__ import annotations

import torch


def l2_normalize(emb: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.norm(emb, p=2, dim=1, keepdim=True).clamp_min(eps)
    return emb / denom


def combine_embeddings(
    ephys: torch.Tensor | None,
    efunc: torch.Tensor | None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    mode: str = "concat",
) -> torch.Tensor:
    if ephys is None and efunc is None:
        raise ValueError("At least one embedding must be provided")
    if ephys is None:
        return beta * efunc
    if efunc is None:
        return alpha * ephys

    if mode == "concat":
        return torch.cat([alpha * ephys, beta * efunc], dim=1)
    if mode == "sum":
        if ephys.shape != efunc.shape:
            raise ValueError("sum mode requires embeddings with matching shapes")
        return alpha * ephys + beta * efunc

    raise ValueError(f"Unsupported combine mode: {mode}")
