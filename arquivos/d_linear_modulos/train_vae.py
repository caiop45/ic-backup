# ──────────────────── main.py ────────────────────
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import optuna
from optuna.trial import TrialState

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer import scale_features
from utils.helpers import decode_hour_from_sincos
from models.vae import VAE                    # —— NOVO —— #

# -------------------------------------------------
# Treino de VAE
# -------------------------------------------------
def treinar_vae(
    X_scaled: np.ndarray,
    input_dim: int,
    hidden_dims=(32, 16, 8),
    latent_dim: int = 4,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
):
    print(f"# DEBUG: Iniciando treino VAE em {device} | "
          f"hidden_dims={hidden_dims} | latent_dim={latent_dim} | "
          f"epochs={epochs} | batch_size={batch_size} | lr={lr}")

    model = VAE(input_dim, hidden_dims, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    ds = TensorDataset(torch.from_numpy(X_scaled))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mse = nn.MSELoss(reduction="mean")

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
        # --- print por época ---
        print(f"[Epoch {epoch:>3}/{epochs}] loss={avg_loss:.6f}")

    print("# DEBUG: Treino VAE finalizado\n")
    return model


# -------------------------------------------------
# Avaliação (MSE de reconstrução em validação)
# -------------------------------------------------
def avaliar_vae(
    model: VAE,
    X_val_scaled: np.ndarray,
    batch_size: int,
    device: str | torch.device = "cpu",
) -> float:
    mse = nn.MSELoss(reduction="mean")
    ds  = TensorDataset(torch.from_numpy(X_val_scaled))
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.no_grad():
        total, n_batches = 0.0, 0
        for (x,) in dl:
            x = x.to(device)
            recon, _, _ = model(x)
            total += mse(recon, x).item()
            n_batches += 1
    val_loss = total / n_batches
    print(f"# DEBUG: Validação concluída | val_loss={val_loss:.6f}")
    return val_loss


# -------------------------------------------------
# Optuna objective
# -------------------------------------------------
def objective(trial, X_train, X_val, input_dim, device):
    # --- hipers ---
    n_layers  = trial.suggest_int("n_layers", 1, 3)
    hidden    = [
        trial.suggest_int(f"units_l{i+1}", 8, 256, step=8)
        for i in range(n_layers)
    ]
    latent_dim = trial.suggest_int("latent_dim", 2, 16)
    epochs     = trial.suggest_int("epochs", 1, 20, step=5)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    print(f"# DEBUG: Trial {trial.number} | params="
          f"layers={n_layers} hidden={hidden} latent={latent_dim} "
          f"epochs={epochs} batch={batch_size} lr={lr:.1e}")

    # --- treino ---
    model = treinar_vae(
        X_scaled=X_train,
        input_dim=input_dim,
        hidden_dims=tuple(hidden),
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    # --- validação ---
    val_loss = avaliar_vae(model, X_val, batch_size, device)

    print(f"# DEBUG: Trial {trial.number} finalizado | val_loss={val_loss:.6f}\n")
    return val_loss


# -------------------------------------------------
# Pipeline principal
# -------------------------------------------------
def process_data():
    print("# DEBUG: Carregando dados reais…")
    (
        dados_reais_orig,
        _,
        _,
        _,
        GMM_FEATURES,
        _,
    ) = load_real_data()

    # -------- separação --------
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    print(f"# DEBUG: gmm_full shape={gmm_full.shape}")

    gmm_train, gmm_val, _ = split_dataset_weekly(
        gmm_full,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    print(f"# DEBUG: gmm_train={len(gmm_train)} | gmm_val={len(gmm_val)}")

    dados_reais_gmm_train = gmm_train[["sin_hr", "cos_hr"]].astype(np.float32)
    dados_reais_gmm_val   = gmm_val[  ["sin_hr", "cos_hr"]].astype(np.float32)

    # -------- scaler --------
    scaler, X_train_scaled = scale_features(dados_reais_gmm_train)
    X_val_scaled           = scaler.transform(dados_reais_gmm_val).astype(np.float32)
    print("# DEBUG: Features escalonadas | "
          f"train_shape={X_train_scaled.shape} val_shape={X_val_scaled.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"# DEBUG: Dispositivo selecionado: {device}")

    # -------- Optuna --------
    print("# DEBUG: Iniciando estudo Optuna…")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda tr: objective(
            tr, X_train_scaled, X_val_scaled, X_train_scaled.shape[1], device
        ),
       n_trials=1,            # ajuste livre – você possui alta capacidade!
        show_progress_bar=True,
    )

    print("\n# --- Melhores hiperparâmetros ---")
    print(study.best_trial.params)

    #-------- Treina VAE final com melhores hiperparâmetros --------
    best = study.best_trial.params
    hidden_dims = tuple(
        best[f"units_l{i+1}"] for i in range(best["n_layers"])
    )

    print("# DEBUG: Treinando VAE final com melhores hiperparâmetros…")


    vae_final = treinar_vae(
        X_scaled=X_train_scaled,
        input_dim=X_train_scaled.shape[1],
        hidden_dims=hidden_dims,
        latent_dim=best["latent_dim"],
        epochs=best["epochs"],
        batch_size=best["batch_size"],
        lr=best["lr"],
        device=device,
    )

    return vae_final, scaler, dados_reais_gmm_train


# -------------------------------------------------
# Amostragem do VAE
# -------------------------------------------------
# ─── substitua TODA a função amostrar_do_vae pelo código abaixo ───
def amostrar_do_vae(
    vae: VAE,
    n_amostras: int,
    scaler,
    *,
    batch_size: int = 100_000,          # lote → ajuste se quiser menos RAM
    device: str | torch.device = "cpu", # "cpu" é à prova de OOM; “cuda” se sobrar VRAM
) -> pd.DataFrame:
    """
    Gera n_amostras usando lotes p/ não estourar memória.
    - Todos os tensores são criados no `device` informado.
    - Após decodificar, cada chunk já volta à CPU como NumPy,
      liberando VRAM imediatamente quando device=’cuda’.
    """
    print(f"# DEBUG: Amostrando {n_amostras:,} pontos (batch={batch_size}) no {device}…")
    vae = vae.to(device).eval()         # garante que o modelo está no mesmo device
    latent_dim = vae.fc_mu.out_features

    chunks = []
    with torch.no_grad():
        for start in range(0, n_amostras, batch_size):
            cur = min(batch_size, n_amostras - start)

            z = torch.randn(cur, latent_dim, device=device)   # (cur, latent_dim)
            decoded = vae.decode(z).cpu().numpy()             # volta à CPU → libera VRAM
            chunks.append(decoded)

            print(f"    · chunk {start + cur:,}/{n_amostras:,} pronto")

    synth = np.concatenate(chunks, axis=0)                    # (n_amostras, n_feats)

    synth_df = pd.DataFrame(
        scaler.inverse_transform(synth),
        columns=["sin_hr", "cos_hr"],
    )
    synth_df["hora_do_dia"] = decode_hour_from_sincos(
        synth_df["sin_hr"], synth_df["cos_hr"]
    )
    print("# DEBUG: Amostragem concluída")
    return synth_df

def comparar_distribuicao_horaria(synth_df: pd.DataFrame) -> None:
    """
    Imprime, para cada hora (0‑23), a contagem de viagens reais (gmm_train),
    a contagem de viagens sintéticas e a diferença percentual:
        %dif = (synth - real) / real * 100
    """
    # --- recarrega só o necessário para obter o gmm_train ---
    dados_reais_orig, _, _, _, GMM_FEATURES, _ = load_real_data()
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()

    gmm_train, _, _ = split_dataset_weekly(
        gmm_full,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )

    # ------ contagens horárias ------
    real_hours = pd.to_datetime(gmm_train["tpep_pickup_datetime"]).dt.hour
    real_counts = real_hours.value_counts().sort_index()

    # “hora_do_dia” já é 0‑23 (int) ou datetime; trata os dois casos
    if np.issubdtype(synth_df["hora_do_dia"].dtype, np.number):
        synth_hours = synth_df["hora_do_dia"].astype(int)
    else:
        synth_hours = pd.to_datetime(synth_df["hora_do_dia"]).dt.hour
    synth_counts = synth_hours.value_counts().sort_index()

    # ------ junta e calcula % diferença ------
    comp = (
        pd.DataFrame({"real": real_counts, "synth": synth_counts})
        .fillna(0)
        .astype(int)
    )
    comp["%dif"] = ((comp["synth"] - comp["real"]) / comp["real"].replace(0, np.nan)) * 100
    comp = comp.round({"%dif": 2})

    print("\n# --------- Distribuição horária: Sintético vs Real ---------")
    print(comp.to_string())
    print("----------------------------------------------------------------\n")

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("# DEBUG: Pipeline principal iniciado\n")
    vae, scaler, dados_reais_gmm_train = process_data()

    # --------- multiplicador sintético ---------
    n_synth = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)
    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")
    print(f"# DEBUG: Gerando {n_synth} linhas sintéticas (multiplier={SYNTHETIC_MULTIPLIER})")

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()  
    synth_df = amostrar_do_vae(vae, n_synth, scaler, device = "cpu", batch_size = 100_000 )

    print("\n# DEBUG: Exemplo de dados sintéticos gerados")
    print(synth_df.head())

    # --------- comparação de distribuição horária ---------
    comparar_distribuicao_horaria(synth_df)

    print("\n# DEBUG: Pipeline concluído com sucesso!")


if __name__ == "__main__":
    main()
