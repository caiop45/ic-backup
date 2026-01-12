import torch
from torch import nn
from typing import Sequence

class VAE(nn.Module):
    """
    Variational Autoencoder genérico para dados tabulares.
    
    Parâmetros
    ----------
    input_dim : int
        Número de features de entrada.
    hidden_dims : Sequence[int], opcional
        Lista com o tamanho de cada camada oculta do encoder.
        O decoder recebe a mesma lista, mas em ordem inversa.
        Ex.: [32, 16, 8] cria encoder 32→16→8 e decoder 8→16→32.
    latent_dim : int, opcional
        Tamanho do vetor latente (gargalo).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (16, 8),
        latent_dim: int = 4,
    ):
        super().__init__()
        
        # ---------- Encoder ----------
        encoder_layers = []
        last_dim = input_dim
        for h in hidden_dims:
            encoder_layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ---------- Decoder (espelho) ----------
        decoder_layers = []
        last_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        decoder_layers.append(nn.Linear(last_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    # ---------- Passos do VAE ----------
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar
