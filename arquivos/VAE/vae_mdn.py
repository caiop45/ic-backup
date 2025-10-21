"""Modelo VAE condicional com cabeça MDN para reconstruir o tempo."""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
import optuna
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from utils.helpers import decode_hour_from_sincos

from mdn_utils import MDNHead, mdn_nll_loss, sample_from_mdn
from od_discretizer import ODDiscretizerArtifacts

__all__ = [
    "VAEMDN",
    "train_vae_mdn",
    "sample_vae_mdn",
    "build_latent_bank",
]


class VAEMDN(nn.Module):
    """VAE condicional em OD com saída MDN para `[sin, cos]` e prior aprendido de hora p(h|OD)."""

    def __init__(
        self,
        pickup_cardinality: int,
        dropoff_cardinality: int,
        *,
        embed_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (128, 64),
        latent_dim: int = 16,
        mdn_components: int = 8,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.pickup_emb = nn.Embedding(pickup_cardinality, embed_dim)
        self.dropoff_emb = nn.Embedding(dropoff_cardinality, embed_dim)

        # Encoder NÃO recebe sin/cos; apenas embeddings de OD, para evitar que z codifique hora
        enc_in = 2 * embed_dim
        encoder_layers: List[nn.Module] = []
        last = enc_in
        for h in hidden_dims:
            encoder_layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        self.encoder = nn.Sequential(*encoder_layers) if encoder_layers else nn.Identity()
        enc_out_dim = last if encoder_layers else enc_in
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_dim, latent_dim)

        # Decoder agora também pode receber condicionamento de hora (sincos_cond com 2 dims)
        dec_in = latent_dim + 2 * embed_dim + 2
        decoder_layers: List[nn.Module] = []
        last_dec = dec_in
        for h in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(last_dec, h), nn.ReLU()])
            last_dec = h
        self.decoder = nn.Sequential(*decoder_layers) if decoder_layers else nn.Identity()
        dec_out_dim = last_dec if decoder_layers else dec_in
        self.mdn_head = MDNHead(dec_out_dim, mdn_components)

        # Cabeça para prior de hora condicional em OD: logits de 24 classes
        self.hour_prior = nn.Sequential(
            nn.Linear(2 * embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 24),
        )

    def encode(
        self, sincos: torch.Tensor, pickup_id: torch.Tensor, dropoff_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pickup_vec = self.pickup_emb(pickup_id)
        dropoff_vec = self.dropoff_emb(dropoff_id)
        # Ignora 'sincos' no encoder para não carregar hora em z
        x = torch.cat([pickup_vec, dropoff_vec], dim=-1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self,
        z: torch.Tensor,
        pickup_id: torch.Tensor,
        dropoff_id: torch.Tensor,
        *,
        sincos_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pickup_vec = self.pickup_emb(pickup_id)
        dropoff_vec = self.dropoff_emb(dropoff_id)
        if sincos_cond is None:
            # Usa vetor nulo quando não houver condicionamento explícito
            sincos_cond = torch.zeros(z.size(0), 2, device=z.device, dtype=z.dtype)
        dec_in = torch.cat([z, pickup_vec, dropoff_vec, sincos_cond], dim=-1)
        h = self.decoder(dec_in)
        return self.mdn_head(h)

    def hour_logits(self, pickup_id: torch.Tensor, dropoff_id: torch.Tensor) -> torch.Tensor:
        pickup_vec = self.pickup_emb(pickup_id)
        dropoff_vec = self.dropoff_emb(dropoff_id)
        od_vec = torch.cat([pickup_vec, dropoff_vec], dim=-1)
        return self.hour_prior(od_vec)

    def forward(
        self, sincos: torch.Tensor, pickup_id: torch.Tensor, dropoff_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(sincos, pickup_id, dropoff_id)
        z = self.reparameterize(mu, logvar)
        pi_logits, mdn_mu, mdn_log_sigma = self.decode(z, pickup_id, dropoff_id, sincos_cond=sincos)
        return pi_logits, mdn_mu, mdn_log_sigma, mu, logvar


def train_vae_mdn(
    df: pd.DataFrame,
    *,
    pickup_cardinality: int,
    dropoff_cardinality: int,
    hidden_dims: Tuple[int, ...],
    latent_dim: int,
    embed_dim: int,
    mdn_components: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device | str,
    trial: Optional[optuna.trial.Trial] = None,
    patience: int = 10,
    min_delta: float = 1e-4,
    kl_beta: float = 1e-3,
    mdn_weight: float = 1.0,
    hour_weight: float = 1.0,
    # Melhorias do prior/decoder (opcionais)
    hour_label_smoothing: float = 0.0,
    hour_smooth_radius: int = 1,
    prior_entropy_bonus: float = 0.0,
    prior_marginal_weight: float = 0.0,
    decoder_cond_mode: str = "gt",  # gt | prior_exp | mix
    decoder_mix_alpha_start: float = 0.0,
    decoder_mix_alpha_end: float = 1.0,
    angle_consistency_weight: float = 0.0,
    # Penalidade leve em sigma para reduzir dispersão angular sem colapsar
    sigma_penalty_weight: float = 0.0,
) -> VAEMDN:
    dataset = TensorDataset(
        torch.tensor(df[["sin_hr", "cos_hr"]].to_numpy(np.float32)),
        torch.tensor(df["pickup_id"].to_numpy(np.int64)),
        torch.tensor(df["dropoff_id"].to_numpy(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VAEMDN(
        pickup_cardinality,
        dropoff_cardinality,
        embed_dim=embed_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        mdn_components=mdn_components,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    # Tabela sin/cos para horas (para expectativa do prior)
    angles_table = torch.arange(24, dtype=torch.float32, device=device) * (2 * np.pi / 24.0)
    sincos_table = torch.stack([torch.sin(angles_table), torch.cos(angles_table)], dim=-1)  # [24,2]

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        last_nll = float("nan")
        last_kld = float("nan")
        last_hour_ce = float("nan")
        # Diagnósticos agregados por época
        total_batches = 0
        total_samples = 0
        sum_entropy = 0.0
        sum_eff_k = 0.0
        sum_max_pi = 0.0
        sum_sigma_mean = 0.0
        sum_mu_norm = 0.0
        hour_hist = np.zeros(24, dtype=np.float64)
        prior_pred_hist = np.zeros(24, dtype=np.float64)
        prior_true_hist = np.zeros(24, dtype=np.float64)
        for sincos, pickup_id, dropoff_id in loader:
            sincos = sincos.to(device)
            pickup_id = pickup_id.to(device)
            dropoff_id = dropoff_id.to(device)

            opt.zero_grad()
            # Encode e amostragem latente
            mu, logvar = model.encode(sincos, pickup_id, dropoff_id)
            z = model.reparameterize(mu, logvar)

            # Prior p(h|OD)
            hour_logits = model.hour_logits(pickup_id, dropoff_id)  # [B,24]
            hour_probs = F.softmax(hour_logits, dim=-1)

            # Alvo de hora (usar arredondamento, coerente com helpers)
            with torch.no_grad():
                ang = torch.remainder(torch.atan2(sincos[:, 0], sincos[:, 1]) + 2 * np.pi, 2 * np.pi)
                hour_float = ang * (24.0 / (2 * np.pi))
                hour_target = torch.clamp(hour_float.round().long(), 0, 23)

            # Label smoothing circular (opcional)
            if hour_label_smoothing > 0.0:
                bsz = hour_logits.size(0)
                y = torch.zeros(bsz, 24, device=device, dtype=torch.float32)
                y.scatter_(1, hour_target.view(-1, 1), 1.0)
                eps = float(hour_label_smoothing)
                R = max(1, int(hour_smooth_radius))
                neigh = torch.zeros_like(y)
                for r in range(1, R + 1):
                    neigh += torch.roll(y, shifts=r, dims=1)
                    neigh += torch.roll(y, shifts=-r, dims=1)
                denom = 2 * R
                y_soft = (1.0 - eps) * y + (eps / max(1, denom)) * neigh
                hour_ce = -(y_soft * F.log_softmax(hour_logits, dim=-1)).sum(dim=-1).mean()
            else:
                hour_ce = F.cross_entropy(hour_logits, hour_target)

            # Expectativa de sincos do prior
            sincos_prior_exp = hour_probs @ sincos_table  # [B,2]

            # Modo de condicionamento do decoder
            mode = (decoder_cond_mode or "gt").lower()
            if mode == "prior_exp":
                alpha = 1.0
            elif mode == "mix":
                if epochs > 1:
                    alpha = float(decoder_mix_alpha_start + (decoder_mix_alpha_end - decoder_mix_alpha_start) * (epoch - 1) / (epochs - 1))
                else:
                    alpha = float(decoder_mix_alpha_end)
                alpha = float(np.clip(alpha, 0.0, 1.0))
            else:
                alpha = 0.0
            sincos_cond = (1.0 - alpha) * sincos + alpha * sincos_prior_exp

            # Decodificação
            pi_logits, mdn_mu, mdn_log_sigma = model.decode(z, pickup_id, dropoff_id, sincos_cond=sincos_cond)

            # Perdas
            nll = mdn_nll_loss(sincos, pi_logits, mdn_mu, mdn_log_sigma, reduction="mean")
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # Entropia média do prior
            ent_prior = -(hour_probs * torch.clamp(hour_probs, min=1e-12).log()).sum(dim=-1).mean()

            # Consistência angular: força a média circular do MDN a seguir sincos_cond
            angle_loss = torch.tensor(0.0, device=device)
            if angle_consistency_weight > 0.0:
                pi_dec = F.softmax(pi_logits, dim=-1)  # [B,K]
                # normaliza mu por componente
                mu_unit = mdn_mu / torch.clamp(torch.linalg.norm(mdn_mu, dim=-1, keepdim=True), min=1e-8)
                v_pred = (pi_dec.unsqueeze(-1) * mu_unit).sum(dim=1)  # [B,2]
                # 1 - cos(sim) entre vetor previsto e sincos_cond
                cos_sim = F.cosine_similarity(v_pred, sincos_cond, dim=-1)
                angle_loss = (1.0 - cos_sim).mean()

            # JSD marginal batch (pred vs. true)
            if prior_marginal_weight > 0.0:
                pred_mass = hour_probs.mean(dim=0)
                with torch.no_grad():
                    true_mass = torch.bincount(hour_target, minlength=24).to(device=device, dtype=torch.float32)
                    true_mass = true_mass / torch.clamp(true_mass.sum(), min=1.0)
                eps = 1e-8
                m = 0.5 * (pred_mass + true_mass)
                jsd_batch = 0.5 * (
                    (pred_mass * (torch.log(torch.clamp(pred_mass, min=eps)) - torch.log(torch.clamp(m, min=eps)))).sum()
                    + (true_mass * (torch.log(torch.clamp(true_mass, min=eps)) - torch.log(torch.clamp(m, min=eps)))).sum()
                )
            else:
                jsd_batch = torch.tensor(0.0, device=device)

            loss = mdn_weight * nll + kl_beta * kld + hour_weight * hour_ce
            if prior_entropy_bonus > 0.0:
                loss = loss - float(prior_entropy_bonus) * ent_prior
            if prior_marginal_weight > 0.0:
                loss = loss + float(prior_marginal_weight) * jsd_batch
            if angle_consistency_weight > 0.0:
                loss = loss + float(angle_consistency_weight) * angle_loss
            # Penalidade de dispersão (σ) opcional
            if sigma_penalty_weight > 0.0:
                sigma = F.softplus(mdn_log_sigma)
                sigma_penalty = (sigma * sigma).mean()
                loss = loss + float(sigma_penalty_weight) * sigma_penalty
            loss.backward()
            opt.step()
            running += loss.item()
            last_nll = nll.item()
            last_kld = kld.item()
            last_hour_ce = hour_ce.item()

            # ---- Diagnósticos por batch ----
            with torch.no_grad():
                pi = F.softmax(pi_logits, dim=-1)  # [B, K]
                # Entropia e K efetivo
                entropy = -(pi * (pi + 1e-12).log()).sum(dim=-1)  # [B]
                eff_k = torch.exp(entropy)  # [B]
                max_pi_batch = pi.max(dim=-1).values  # [B]

                # Sigma médio (garantindo positividade)
                sigma = F.softplus(mdn_log_sigma)  # [B, K, 2]
                sigma_mean = sigma.mean()

                # Norma média de mu e histograma horário ponderado por pi
                mu = mdn_mu  # [B, K, 2]
                mu_norm = torch.linalg.norm(mu, dim=-1).mean()  # média de ||mu||
                # Direção -> hora [0,24)
                # Evita divisão por zero na normalização (usamos atan2 diretamente)
                hours = torch.remainder(torch.atan2(mu[..., 0], mu[..., 1]) + 2 * np.pi, 2 * np.pi) * (24.0 / (2 * np.pi))  # [B, K]
                # Acúmulo do histograma ponderado por pi
                hours_np = hours.detach().cpu().numpy().reshape(-1)
                weights_np = pi.detach().cpu().numpy().reshape(-1)
                bins = np.clip(np.floor(hours_np).astype(int), 0, 23)
                np.add.at(hour_hist, bins, weights_np)

                # Prior de hora: acumula distribuição predita e verdadeira
                prior_probs = hour_probs.detach().cpu().numpy()
                prior_bins_true = hour_target.detach().cpu().numpy()
                prior_pred_hist += prior_probs.sum(axis=0)
                for b in prior_bins_true:
                    prior_true_hist[int(b)] += 1.0

                bsz = sincos.size(0)
                total_batches += 1
                total_samples += bsz
                sum_entropy += entropy.mean().item()
                sum_eff_k += eff_k.mean().item()
                sum_max_pi += max_pi_batch.mean().item()
                sum_sigma_mean += sigma_mean.item()
                sum_mu_norm += mu_norm.item()

        avg_loss = running / max(len(loader), 1)
        print(
            f"[MDN][Epoch {epoch:>3}/{epochs}] loss={avg_loss:.6f} "
            f"(nll={last_nll:.6f}, kl={last_kld:.6f}, hour_ce={last_hour_ce:.6f})"
        )

        # ---- Prints de diagnóstico por época ----
        if total_batches > 0:
            hour_mass = hour_hist / max(hour_hist.sum(), 1e-12)
            peak_hour = int(np.argmax(hour_mass))
            mass_madrugada = hour_mass[0:4].sum()
            mass_tarde = hour_mass[15:19].sum()
            mass_noite = hour_mass[20:24].sum()
            print(
                "[MDN][Diag] "
                f"effK_med={sum_eff_k/total_batches:.2f} | "
                f"maxPi_med={sum_max_pi/total_batches:.3f} | "
                f"entropy_med={sum_entropy/total_batches:.3f} | "
                f"sigma_med={sum_sigma_mean/total_batches:.3f} | "
                f"||mu||_med={sum_mu_norm/total_batches:.3f}"
            )
            print(
                "[MDN][Diag] Horas: pico=", peak_hour,
                f"| madrugada(0-3)={mass_madrugada:.3f}",
                f"| tarde(15-18)={mass_tarde:.3f}",
                f"| noite(20-23)={mass_noite:.3f}"
            )

            # Diagnóstico do prior p(h|OD)
            pred_mass = prior_pred_hist / max(prior_pred_hist.sum(), 1e-12)
            true_mass = prior_true_hist / max(prior_true_hist.sum(), 1e-12)
            eps = 1e-12
            m = 0.5 * (pred_mass + true_mass)
            jsd = 0.5 * (
                (true_mass * (np.log(true_mass + eps) - np.log(m + eps))).sum()
                + (pred_mass * (np.log(pred_mass + eps) - np.log(m + eps))).sum()
            )
            peak_pred = int(np.argmax(pred_mass)) if pred_mass.size else -1
            peak_true = int(np.argmax(true_mass)) if true_mass.size else -1
            print(
                f"[MDN][Prior h|OD] JSD={jsd:.4f} peak_pred={peak_pred} peak_true={peak_true} "
                f"std_pred={pred_mass.std():.4f} std_true={true_mass.std():.4f}"
            )

        if trial is not None:
            trial.report(avg_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"[MDN] Early stopping acionado após {patience} épocas sem melhora."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def sample_vae_mdn(
    model: VAEMDN,
    *,
    n_samples: int,
    od_artifacts: ODDiscretizerArtifacts,
    od_distribution: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 100_000,
    latent_bank: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    # Ajustes de prior de hora (opcionais)
    hour_temperature: float = 1.0,
    hour_bias: np.ndarray | None = None,
    hour_blend_alpha: float = 0.0,
    global_hour_hist: np.ndarray | None = None,
    # Se True, ignora MDN e usa diretamente o prior para definir sin/cos
    use_prior_hours_directly: bool = False,
    # Se True, usa a média do MDN (média circular ponderada por π) ao invés de amostragem
    use_mdn_mean: bool = False,
) -> pd.DataFrame:
    model = model.to(device).eval()

    latent_dim = model.latent_dim
    od_distribution = od_distribution.astype(np.float64)
    od_distribution /= od_distribution.sum()

    od_to_pickup = np.asarray(od_artifacts.od_to_pickup_id, dtype=np.int64)
    od_to_dropoff = np.asarray(od_artifacts.od_to_dropoff_id, dtype=np.int64)

    sin_list: List[np.ndarray] = []
    cos_list: List[np.ndarray] = []
    hour_list: List[np.ndarray] = []
    od_ids: List[np.ndarray] = []
    pickup_ids: List[np.ndarray] = []
    dropoff_ids: List[np.ndarray] = []

    for start in range(0, n_samples, batch_size):
        cur = min(batch_size, n_samples - start)
        sampled_od = np.random.choice(
            np.arange(len(od_distribution)), size=cur, p=od_distribution
        ).astype(np.int64)
        pickup = od_to_pickup[sampled_od]
        dropoff = od_to_dropoff[sampled_od]

        if latent_bank is not None:
            z_np = np.empty((cur, latent_dim), dtype=np.float32)
            for i, od_id in enumerate(sampled_od):
                entry = latent_bank.get(int(od_id))
                if entry is None:
                    z_np[i] = np.random.randn(latent_dim).astype(np.float32)
                    continue
                mu_store, logvar_store = entry
                if len(mu_store) == 0:
                    z_np[i] = np.random.randn(latent_dim).astype(np.float32)
                    continue
                j = np.random.randint(len(mu_store))
                mu_vec = mu_store[j]
                logvar_vec = logvar_store[j]
                sigma_vec = np.exp(0.5 * logvar_vec)
                z_np[i] = mu_vec + sigma_vec * np.random.randn(latent_dim)
            z = torch.from_numpy(z_np).to(device)
        else:
            z = torch.randn(cur, latent_dim, device=device)
        pickup_t = torch.from_numpy(pickup).to(device)
        dropoff_t = torch.from_numpy(dropoff).to(device)

        # Amostra horas do prior p(h|OD) com ajustes opcionais
        hour_logits = model.hour_logits(pickup_t, dropoff_t)  # [cur, 24]
        logits = hour_logits
        # Aplica viés global por hora (em log-prob space)
        if hour_bias is not None:
            hb = torch.from_numpy(hour_bias.astype(np.float32)).to(logits.device)
            if hb.numel() != 24:
                raise ValueError("hour_bias deve ter shape (24,)")
            logits = logits + hb.view(1, 24)
        # Aplica temperatura
        if hour_temperature and abs(hour_temperature - 1.0) > 1e-6:
            logits = logits / float(hour_temperature)
        hour_probs = F.softmax(logits, dim=-1)
        # Blend opcional com histograma global real
        if hour_blend_alpha and hour_blend_alpha > 0.0 and global_hour_hist is not None:
            gh = torch.from_numpy(global_hour_hist.astype(np.float64)).to(logits.device, dtype=torch.float32)
            gh = gh / max(float(gh.sum().item()), 1e-12)
            hour_probs = (1.0 - float(hour_blend_alpha)) * hour_probs + float(hour_blend_alpha) * gh.view(1, 24)
            # Renormaliza por segurança
            hour_probs = hour_probs / torch.clamp(hour_probs.sum(dim=-1, keepdim=True), min=1e-12)
        # Amostragem categórica vetorizada
        cat = torch.distributions.Categorical(probs=hour_probs)
        hours_t = cat.sample()
        hours = hours_t.detach().cpu().numpy().astype(np.int64)
        # Constrói sincos condicionais a partir de horas
        angles = hours * (2 * np.pi / 24.0)
        sincos_cond = torch.stack(
            [torch.from_numpy(np.sin(angles)).to(device, dtype=torch.float32),
             torch.from_numpy(np.cos(angles)).to(device, dtype=torch.float32)], dim=1
        )  # [cur, 2]

        if use_prior_hours_directly:
            # Usa diretamente as horas amostradas do prior para definir sin/cos
            sin_vals = np.sin(angles).astype(np.float32)
            cos_vals = np.cos(angles).astype(np.float32)
            sin_list.append(sin_vals)
            cos_list.append(cos_vals)
        else:
            pi_logits, mdn_mu, mdn_log_sigma = model.decode(
                z, pickup_t, dropoff_t, sincos_cond=sincos_cond
            )
            if use_mdn_mean:
                pi = F.softmax(pi_logits, dim=-1)
                mu_unit = mdn_mu / torch.clamp(torch.linalg.norm(mdn_mu, dim=-1, keepdim=True), min=1e-8)
                vec = (pi.unsqueeze(-1) * mu_unit).sum(dim=1)
                vec = vec / torch.clamp(vec.norm(dim=-1, keepdim=True), min=1e-8)
                samples = vec.cpu().numpy()
            else:
                samples = sample_from_mdn(pi_logits, mdn_mu, mdn_log_sigma).cpu().numpy()
            sin_list.append(samples[:, 0])
            cos_list.append(samples[:, 1])
        hour_list.append(hours)
        od_ids.append(sampled_od)
        pickup_ids.append(pickup)
        dropoff_ids.append(dropoff)

    sin = np.concatenate(sin_list)
    cos = np.concatenate(cos_list)
    od_array = np.concatenate(od_ids)
    pickup_array = np.concatenate(pickup_ids)
    dropoff_array = np.concatenate(dropoff_ids)
    hours_array = np.concatenate(hour_list)

    pickup_location = [od_artifacts.id_to_pickup(int(idx)) for idx in pickup_array]
    dropoff_location = [od_artifacts.id_to_dropoff(int(idx)) for idx in dropoff_array]
    od_pair = [f"{p}-{d}" for p, d in zip(pickup_location, dropoff_location)]

    df = pd.DataFrame(
        {
            "sin_hr": sin,
            "cos_hr": cos,
            "pickup_id": pickup_array,
            "dropoff_id": dropoff_array,
            "od_id": od_array,
            "hora_target": hours_array,
            "pickup_location": pickup_location,
            "dropoff_location": dropoff_location,
            "od_pair": od_pair,
        }
    )
    df["hora_do_dia"] = decode_hour_from_sincos(df["sin_hr"], df["cos_hr"])
    return df


@torch.no_grad()
def build_latent_bank(
    model: VAEMDN,
    df: pd.DataFrame,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 8192,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Constrói um banco de representações latentes dos dados de treino agrupadas por par OD.
    
    Este banco é usado durante a geração para amostrar a partir de distribuições latentes
    observadas nos dados reais, em vez de usar amostras puramente aleatórias.
    Isso melhora o realismo das amostras sintéticas, mas pode reduzir a diversidade.
    
    Returns:
        dict: {od_id: (μ_arrays, logvar_arrays)} - representações latentes por par OD
    """
    # Coloca modelo em modo de avaliação (sem gradientes)
    model = model.to(device).eval()

    # Converte dados do DataFrame para tensores PyTorch
    sincos_tensor = torch.tensor(df[["sin_hr", "cos_hr"]].to_numpy(np.float32))
    pickup_tensor = torch.tensor(df["pickup_id"].to_numpy(np.int64))
    dropoff_tensor = torch.tensor(df["dropoff_id"].to_numpy(np.int64))
    od_tensor = torch.tensor(df["od_id"].to_numpy(np.int64))

    # Cria DataLoader para processamento em batches
    dataset = TensorDataset(sincos_tensor, pickup_tensor, dropoff_tensor, od_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Dicionários para armazenar representações latentes agrupadas por od_id
    mu_store: dict[int, list[np.ndarray]] = {}
    logvar_store: dict[int, list[np.ndarray]] = {}

    # Processa todos os dados através do encoder
    for sincos, pickup, dropoff, od in loader:
        # Move tensores para dispositivo de computação
        sincos = sincos.to(device)
        pickup = pickup.to(device)
        dropoff = dropoff.to(device)
        
        # Extrai representações latentes (μ, log σ) através do encoder
        mu, logvar = model.encode(sincos, pickup, dropoff)
        
        # Move resultados de volta para CPU como arrays numpy
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
        od_np = od.numpy()
        
        # Agrupa representações latentes por od_id
        for idx, od_id in enumerate(od_np):
            # Adiciona μ e log σ específicos desta amostra ao banco do respectivo od_id
            mu_store.setdefault(int(od_id), []).append(mu[idx])
            logvar_store.setdefault(int(od_id), []).append(logvar[idx])

    # Converte listas de arrays para arrays numpy empilhados
    latent_bank: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for od_id, mu_list in mu_store.items():
        logvar_list = logvar_store[od_id]
        # Empilha todos os μ e log σ deste od_id em arrays 2D
        latent_bank[od_id] = (
            np.stack(mu_list).astype(np.float32),  # Shape: (n_samples_od, latent_dim)
            np.stack(logvar_list).astype(np.float32),  # Shape: (n_samples_od, latent_dim)
        )

    return latent_bank
