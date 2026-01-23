"""Conditional rational-quadratic spline flow on [0, 1].

Implements a monotonic 1D spline with exact log-likelihood via change of variables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SplineConfig:
    context_dim: int
    num_layers: int = 5
    num_bins: int = 8
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    eps: float = 1e-6


class _SplineLayer1D(nn.Module):
    def __init__(
        self,
        *,
        context_dim: int,
        num_bins: int,
        min_bin_width: float,
        min_bin_height: float,
        min_derivative: float,
    ) -> None:
        super().__init__()
        if num_bins <= 1:
            raise ValueError("num_bins must be > 1")
        self.num_bins = int(num_bins)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)

        if self.min_bin_width * self.num_bins >= 1.0:
            raise ValueError("min_bin_width * num_bins must be < 1")
        if self.min_bin_height * self.num_bins >= 1.0:
            raise ValueError("min_bin_height * num_bins must be < 1")

        hidden = max(32, context_dim)
        out_dim = 3 * self.num_bins + 1
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _compute_params(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.net(context)
        k = self.num_bins
        widths_raw = params[:, :k]
        heights_raw = params[:, k : 2 * k]
        derivatives_raw = params[:, 2 * k :]

        widths = F.softmax(widths_raw, dim=1)
        widths = self.min_bin_width + (1.0 - self.min_bin_width * k) * widths

        heights = F.softmax(heights_raw, dim=1)
        heights = self.min_bin_height + (1.0 - self.min_bin_height * k) * heights

        derivatives = self.min_derivative + F.softplus(derivatives_raw)

        cumwidths = F.pad(torch.cumsum(widths, dim=1), (1, 0), value=0.0)
        cumheights = F.pad(torch.cumsum(heights, dim=1), (1, 0), value=0.0)
        cumwidths[:, -1] = 1.0
        cumheights[:, -1] = 1.0

        return widths, heights, derivatives, cumwidths, cumheights

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivatives, cumwidths, cumheights = self._compute_params(context)
        x = x.unsqueeze(1)

        bin_idx = torch.sum(x >= cumwidths[:, 1:], dim=1)
        bin_idx = torch.clamp(bin_idx, max=self.num_bins - 1)

        idx = bin_idx.unsqueeze(1)
        x0 = cumwidths.gather(1, idx).squeeze(1)
        x1 = cumwidths.gather(1, idx + 1).squeeze(1)
        y0 = cumheights.gather(1, idx).squeeze(1)
        y1 = cumheights.gather(1, idx + 1).squeeze(1)

        w = widths.gather(1, idx).squeeze(1)
        h = heights.gather(1, idx).squeeze(1)
        d0 = derivatives.gather(1, idx).squeeze(1)
        d1 = derivatives.gather(1, idx + 1).squeeze(1)

        theta = (x.squeeze(1) - x0) / w
        s = h / w
        a = d0 + d1 - 2.0 * s

        numerator = h * (s * theta * theta + d0 * theta * (1.0 - theta))
        denominator = s + a * theta * (1.0 - theta)
        y = y0 + numerator / denominator

        derivative = (
            s * s
            * (d1 * theta * theta + 2.0 * s * theta * (1.0 - theta) + d0 * (1.0 - theta) ** 2)
            / (denominator * denominator)
        )
        logabsdet = torch.log(derivative.clamp_min(1e-12))
        return y, logabsdet

    def inverse(
        self, y: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivatives, cumwidths, cumheights = self._compute_params(context)
        y = y.unsqueeze(1)

        bin_idx = torch.sum(y >= cumheights[:, 1:], dim=1)
        bin_idx = torch.clamp(bin_idx, max=self.num_bins - 1)
        idx = bin_idx.unsqueeze(1)

        x0 = cumwidths.gather(1, idx).squeeze(1)
        x1 = cumwidths.gather(1, idx + 1).squeeze(1)
        y0 = cumheights.gather(1, idx).squeeze(1)
        y1 = cumheights.gather(1, idx + 1).squeeze(1)

        w = widths.gather(1, idx).squeeze(1)
        h = heights.gather(1, idx).squeeze(1)
        d0 = derivatives.gather(1, idx).squeeze(1)
        d1 = derivatives.gather(1, idx + 1).squeeze(1)

        y_hat = (y.squeeze(1) - y0) / h
        s = h / w
        a = d0 + d1 - 2.0 * s

        A = (s - d0) + y_hat * a
        B = d0 - y_hat * a
        C = -y_hat * s

        eps = 1e-12
        linear = torch.abs(A) < eps
        theta = torch.zeros_like(y_hat)
        theta_linear = -C / B
        theta = torch.where(linear, theta_linear, theta)

        disc = B * B - 4.0 * A * C
        disc = torch.clamp(disc, min=0.0)
        sqrt_disc = torch.sqrt(disc)

        theta_quadratic = (2.0 * C) / (-B - sqrt_disc)
        theta = torch.where(linear, theta, theta_quadratic)
        theta = theta.clamp(0.0, 1.0)

        x = x0 + theta * w

        denominator = s + a * theta * (1.0 - theta)
        derivative = (
            s * s
            * (d1 * theta * theta + 2.0 * s * theta * (1.0 - theta) + d0 * (1.0 - theta) ** 2)
            / (denominator * denominator)
        )
        logabsdet = -torch.log(derivative.clamp_min(1e-12))
        return x, logabsdet


class ConditionalSplineFlow1D(nn.Module):
    """Conditional spline flow on [0,1] for residual time r."""

    def __init__(self, *, context_dim: int, num_layers: int = 5, num_bins: int = 8,
                 min_bin_width: float = 1e-3, min_bin_height: float = 1e-3,
                 min_derivative: float = 1e-3, eps: float = 1e-6) -> None:
        super().__init__()
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.context_dim = int(context_dim)
        self.num_layers = int(num_layers)
        self.num_bins = int(num_bins)
        self.eps = float(eps)

        self.layers = nn.ModuleList(
            [
                _SplineLayer1D(
                    context_dim=self.context_dim,
                    num_bins=self.num_bins,
                    min_bin_width=min_bin_width,
                    min_bin_height=min_bin_height,
                    min_derivative=min_derivative,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self, z: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_logabsdet = torch.zeros_like(z)
        for layer in self.layers:
            x, logabsdet = layer.forward(x, context)
            total_logabsdet = total_logabsdet + logabsdet
        return x, total_logabsdet

    def inverse(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_logabsdet = torch.zeros_like(x)
        for layer in reversed(self.layers):
            z, logabsdet = layer.inverse(z, context)
            total_logabsdet = total_logabsdet + logabsdet
        return z, total_logabsdet

    def log_prob(self, r: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if r.dim() != 1:
            raise ValueError("r must be a 1D tensor")
        if context.dim() != 2:
            raise ValueError("context must be a 2D tensor")
        if r.numel() != context.shape[0]:
            raise ValueError("r and context batch sizes must match")

        r_clamped = r.clamp(self.eps, 1.0 - self.eps)
        inside = torch.isfinite(r) & (r >= self.eps) & (r <= 1.0 - self.eps)
        _, logabsdet = self.inverse(r_clamped, context)
        log_prob = logabsdet
        log_prob = torch.where(inside, log_prob, torch.full_like(log_prob, float("-inf")))
        return log_prob

    def sample(self, context: torch.Tensor, *, seed: int | None = None) -> torch.Tensor:
        if context.dim() != 2:
            raise ValueError("context must be a 2D tensor")
        device = context.device
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        z = torch.rand(context.shape[0], device=device, generator=generator)
        z = z * (1.0 - 2.0 * self.eps) + self.eps
        x, _ = self.forward(z, context)
        return x.clamp(self.eps, 1.0 - self.eps)
