# ──────────────────── main.py ────────────────────
from __future__ import annotations

import sys
import contextlib
from pathlib import Path
from typing import Iterable

import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.pruners import MedianPruner

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from utils.helpers import decode_hour_from_sincos
from models.vae import VAE


# -------------------------------------------------
# Utilitário: duplica stdout/stderr (console + arquivo)
# -------------------------------------------------
class Tee:
    """Escreve simultaneamente em diversos streams (ex.: console e arquivo)."""

    def __init__(self, *streams: Iterable):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # garante que a saída apareça em tempo‑real

    def flush(self):
        for s in self.streams:
            s.flush()


# -------------------------------------------------
# Treino de VAE (com early stopping + report p/ pruning)
# -------------------------------------------------
def treinar_vae(
    X: np.ndarray,
    input_dim: int,
    trial: optuna.trial.Trial | None,
    *,
    hidden_dims=(32, 16, 8),
    latent_dim: int = 4,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
    # early‑stopping
    patience: int = 10,
    min_delta: float = 1e-4,
):
    print(
        f"# DEBUG: Iniciando treino VAE em {device} | "
        f"hidden_dims={hidden_dims} | latent_dim={latent_dim} | "
        f"epochs={epochs} | batch_size={batch_size} | lr={lr}"
    )

    model = VAE(input_dim, hidden_dims, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="mean")

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    best_loss = float("inf")
    best_state = None
    epochs_since_improve = 0

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(x)
            recon_loss = mse(recon, x)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-3 * kld
            loss.backward()
            opt.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dl)
        print(f"[Epoch {epoch:>3}/{epochs}] recon+kld={avg_loss:.6f}")

        # --- Optuna pruning ---
        if trial is not None:
            trial.report(avg_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # --- Early stopping ---
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            print(
                f"# DEBUG: Early stopping disparado – "
                f"sem melhora por {patience} épocas.\n"
            )
            break

    # restaura melhor estado
    if best_state is not None:
        model.load_state_dict(best_state)

    print("# DEBUG: Treino VAE finalizado\n")
    return model


# -------------------------------------------------
# Métrica: maior diferença percentual absoluta (pontos‑percentuais)
# -------------------------------------------------
def max_abs_percent_rel_diff(
    synth_df: pd.DataFrame, gmm_train: pd.DataFrame
) -> float:
    real_probs = (
        pd.to_datetime(gmm_train["tpep_pickup_datetime"])
        .dt.hour.value_counts(normalize=True)
        .sort_index()
    )
    synth_probs = (
        synth_df["hora_do_dia"]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
    )

    all_hours = pd.RangeIndex(24)
    real_probs = real_probs.reindex(all_hours, fill_value=0)
    synth_probs = synth_probs.reindex(all_hours, fill_value=0)

    diff_pct = (
        (synth_probs - real_probs).abs()
        / real_probs.replace(0, np.nan).fillna(1e-9)
        * 100
    )
    return diff_pct.max()


# -------------------------------------------------
# Amostragem do VAE (sem scaler) + renormaliza círculo
# -------------------------------------------------
def amostrar_do_vae(
    vae: VAE,
    n_amostras: int,
    *,
    batch_size: int = 100_000,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    print(
        f"# DEBUG: Amostrando {n_amostras:,} pontos (batch={batch_size}) no {device}…"
    )
    vae = vae.to(device).eval()
    latent_dim = vae.fc_mu.out_features
    chunks = []

    with torch.no_grad():
        for start in range(0, n_amostras, batch_size):
            cur = min(batch_size, n_amostras - start)
            z = torch.randn(cur, latent_dim, device=device)
            decoded = vae.decode(z).cpu().numpy()
            chunks.append(decoded)
            print(f"    · chunk {start + cur:,}/{n_amostras:,} pronto")

    synth = np.concatenate(chunks, axis=0)
    synth_df = pd.DataFrame(synth, columns=["sin_hr", "cos_hr"])

    # renormaliza para a circunferência unitária
    r = np.sqrt(synth_df["sin_hr"] ** 2 + synth_df["cos_hr"] ** 2)
    synth_df["sin_hr"] /= r
    synth_df["cos_hr"] /= r

    synth_df["hora_do_dia"] = decode_hour_from_sincos(
        synth_df["sin_hr"], synth_df["cos_hr"]
    )
    print("# DEBUG: Amostragem concluída")
    return synth_df


# -------------------------------------------------
# Optuna objective
# -------------------------------------------------
def objective(trial, gmm_train, X_train, input_dim, device):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden = [
        trial.suggest_int(f"units_l{i+1}", 16, 32, step=16) for i in range(n_layers)
    ]#fazer o suggest_categorial usando o exponencial 16, 32, 64)
    latent_dim = trial.suggest_int("latent_dim", 2, 16)
    epochs = trial.suggest_int("epochs", 50, 200, step=50)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    print(
        f"# DEBUG: Trial {trial.number} | params="
        f"layers={n_layers} hidden={hidden} latent={latent_dim} "
        f"epochs={epochs} batch={batch_size} lr={lr:.1e}"
    )

    model = treinar_vae(
        X_train,
        input_dim,
        trial,
        hidden_dims=tuple(hidden),
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    synth_df = amostrar_do_vae(
        model,
        n_amostras=80_000,  # amostragem “rápida” só p/ métrica
        device=device,
    )
    metric = max_abs_percent_rel_diff(synth_df, gmm_train)
    print(f"# DEBUG: Trial {trial.number} finalizado | max|Δ|={metric:.2f} pp\n")
    return metric


# -------------------------------------------------
# Pipeline principal
# -------------------------------------------------
def process_data():
    print("# DEBUG: Carregando dados reais…")
    dados_reais_orig, *_ , GMM_FEATURES, _ = load_real_data()

    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, _ = split_dataset_weekly(
        gmm_full,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    print(f"# DEBUG: gmm_train={len(gmm_train)} | gmm_val={len(gmm_val)}")

    # sem scaling
    X_train = gmm_train[["sin_hr", "cos_hr"]].to_numpy(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# DEBUG: Dispositivo selecionado: {device}")

    # Optuna
    print("# DEBUG: Iniciando estudo Optuna…")
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_warmup_steps=1),
    )
    study.optimize(
        lambda tr: objective(tr, gmm_train, X_train, X_train.shape[1], device),
        n_trials=30,
        show_progress_bar=True,
    )

    print("\n# --- Melhores hiperparâmetros ---")
    print(study.best_trial.params)
    best = study.best_trial.params
    hidden_dims = tuple(best[f"units_l{i+1}"] for i in range(best["n_layers"]))

    # Treina VAE final
    print("# DEBUG: Treinando VAE final com melhores hiperparâmetros…")
    vae_final = treinar_vae(
        X_train,
        X_train.shape[1],
        trial=None,
        hidden_dims=hidden_dims,
        latent_dim=best["latent_dim"],
        epochs=best["epochs"],
        batch_size=best["batch_size"],
        lr=best["lr"],
        device=device,
    )

    return vae_final, gmm_train


# -------------------------------------------------
# Relatório detalhado
# -------------------------------------------------
def comparar_distribuicao_horaria(
    synth_df: pd.DataFrame, gmm_train: pd.DataFrame
) -> None:
    real_counts = (
        pd.to_datetime(gmm_train["tpep_pickup_datetime"])
        .dt.hour.value_counts()
        .sort_index()
    )
    synth_counts = (
        synth_df["hora_do_dia"].astype(int).value_counts().sort_index()
    )

    real_probs = (real_counts / real_counts.sum()) * 100
    synth_probs = (synth_counts / synth_counts.sum()) * 100

    comp = (
        pd.DataFrame({"real_%": real_probs, "synth_%": synth_probs})
        .fillna(0)
        .round(2)
    )
    comp["Δ (pp)"] = (comp["synth_%"] - comp["real_%"]).round(2)

    print("\n# --------- Distribuição horária (% do total) ---------")
    print(comp.to_string())
    print("Max |Δ| =", comp["Δ (pp)"].abs().max(), "pp")
    print("-------------------------------------------------------\n")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("# DEBUG: Pipeline principal iniciado\n")
    vae, gmm_train = process_data()

    n_synth = int(len(gmm_train) * SYNTHETIC_MULTIPLIER)
    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")
    print(
        f"# DEBUG: Gerando {n_synth} linhas sintéticas "
        f"(multiplier={SYNTHETIC_MULTIPLIER})"
    )

    torch.cuda.empty_cache()
    synth_df = amostrar_do_vae(
        vae,
        n_amostras=n_synth,
        device="cpu",
        batch_size=100_000,
    )

    print("\n# DEBUG: Exemplo de dados sintéticos gerados")
    print(synth_df.head())

    comparar_distribuicao_horaria(synth_df, gmm_train)
    print("\n# DEBUG: Pipeline concluído com sucesso!")


# -------------------------------------------------
# Entrada + log para .txt (console + arquivo simultâneos)
# -------------------------------------------------
if __name__ == "__main__":
    log_path = Path(__file__).with_name("train_vae_log.txt")

    # Mantemos referências aos streams originais antes do redirecionamento
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with log_path.open("w") as f:
        # Cria os duplicadores
        tee_out = Tee(orig_stdout, f)
        tee_err = Tee(orig_stdout, f)  # stderr também para stdout no console

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(
            tee_err
        ):
            main()

    # Mensagem final (apenas no console)
    print(
        f"Execução concluída. Todo o output foi salvo em '{log_path.name}'."
    )
