from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last_dim, h))
        layers.append(nn.ReLU())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class TVAEAutoregressive(nn.Module):
    def __init__(
        self,
        *,
        column_sizes: Dict[str, int],
        order: List[str],
        encoder_hidden_dims: Iterable[int] = (256, 128),
        decoder_hidden_dims: Iterable[int] = (128, 128),
        latent_dim: int = 32,
    ) -> None:
        super().__init__()

        self.column_sizes = {k: int(v) for k, v in column_sizes.items()}
        self.order = list(order)
        self.encoder_hidden_dims = tuple(encoder_hidden_dims)
        self.decoder_hidden_dims = tuple(decoder_hidden_dims)
        self.latent_dim = int(latent_dim)

        input_dim = sum(self.column_sizes.values())
        self.input_dim = int(input_dim)

        encoder_layers: List[nn.Module] = []
        last_dim = input_dim
        for h in self.encoder_hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h))
            encoder_layers.append(nn.ReLU())
            last_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(last_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(last_dim, self.latent_dim)

        self.heads = nn.ModuleDict()
        context_dim = 0
        for col in self.order:
            head_in = self.latent_dim + context_dim
            self.heads[col] = _build_mlp(head_in, self.decoder_hidden_dims, self.column_sizes[col])
            context_dim += self.column_sizes[col]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_teacher_forcing(
        self, z: torch.Tensor, y_indices: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        logits: Dict[str, torch.Tensor] = {}
        context: torch.Tensor | None = None

        for col in self.order:
            head = self.heads[col]
            if context is None:
                head_in = z
            else:
                head_in = torch.cat([z, context], dim=1)
            logits[col] = head(head_in)

            idx = y_indices[col]
            one_hot = F.one_hot(idx, num_classes=self.column_sizes[col]).float()
            context = one_hot if context is None else torch.cat([context, one_hot], dim=1)

        return logits

    def sample(
        self, z: torch.Tensor, *, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        samples: Dict[str, torch.Tensor] = {}
        context: torch.Tensor | None = None

        for col in self.order:
            head = self.heads[col]
            if context is None:
                head_in = z
            else:
                head_in = torch.cat([z, context], dim=1)

            logits = head(head_in)
            if temperature != 1.0:
                logits = logits / float(temperature)
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            samples[col] = idx

            one_hot = F.one_hot(idx, num_classes=self.column_sizes[col]).float()
            context = one_hot if context is None else torch.cat([context, one_hot], dim=1)

        return samples

    def forward(
        self, x: torch.Tensor, y_indices: Dict[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode_teacher_forcing(z, y_indices)
        return logits, mu, logvar
