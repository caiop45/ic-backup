"""Componentes reutilizáveis para cabeçalhos MDN 2D."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MDNHead",
    "mdn_nll_loss",
    "sample_from_mdn",
]


class MDNHead(nn.Module):
    """Mixture Density Network head para reconstruir vetores 2D."""

    def __init__(self, in_dim: int, n_components: int) -> None:
        super().__init__()
        self.n_components = n_components
        self.logits = nn.Linear(in_dim, n_components)
        self.mu = nn.Linear(in_dim, n_components * 2)
        self.log_sigma = nn.Linear(in_dim, n_components * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits = self.logits(x)
        mu = self.mu(x).view(-1, self.n_components, 2)
        log_sigma = self.log_sigma(x).view(-1, self.n_components, 2)
        return pi_logits, mu, log_sigma


def _mdn_component_log_prob(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = F.softplus(log_sigma) + 1e-4
    diff = (target.unsqueeze(1) - mu) / sigma
    log_det = torch.log(sigma).sum(dim=-1)
    quad = 0.5 * (diff.pow(2).sum(dim=-1))
    norm_const = target.shape[-1] * 0.5 * math.log(2 * math.pi)
    return -(quad + log_det + norm_const)


def mdn_nll_loss(
    target: torch.Tensor,
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calcula o NLL de um MDN 2D."""

    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_prob = torch.logsumexp(
        log_pi + _mdn_component_log_prob(target, mu, log_sigma), dim=-1
    )
    nll = -log_prob
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll


@torch.no_grad()
def sample_from_mdn(
    pi_logits: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Amostra vetores 2D de um MDN e normaliza para raio unitário."""

    pi = F.softmax(pi_logits, dim=-1)
    sigma = F.softplus(log_sigma) + 1e-4
    comp = torch.distributions.Categorical(pi).sample()
    idx = torch.arange(pi.shape[0], device=pi_logits.device)
    mu_sel = mu[idx, comp]
    sigma_sel = sigma[idx, comp]
    eps = torch.randn_like(mu_sel)
    sample = mu_sel + sigma_sel * eps
    norm = torch.clamp(sample.norm(dim=-1, keepdim=True), min=1e-6)
    return sample / norm
