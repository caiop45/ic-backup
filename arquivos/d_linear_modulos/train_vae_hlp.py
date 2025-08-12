"""Treino de VAE com métrica baseada em matriz origem-destino (OD).

Diferenças p/ train_vae.py:
* Optuna otimiza a maior diferença absoluta entre as distribuições OD
  (top-10 pares) dos dados sintéticos e reais.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import contextlib
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from utils.zone_id import assign_zone_names
from train_vae import LOCATION_COLS, Tee, amostrar_do_vae, treinar_vae

# --------------------------------------------------------------------------- #
#                       MÉTRICAS DE ORIGEM-DESTINO (OD)                       #
# --------------------------------------------------------------------------- #


def real_od_distribution(df: pd.DataFrame) -> tuple[pd.Series, pd.Index]:
    """Retorna distribuição OD (top-10 pares) e os pares considerados."""
    df_zones = assign_zone_names(df.copy())
    counts = df_zones.groupby(["PU_zone_name", "DO_zone_name"]).size()
    top_pairs = counts.sort_values(ascending=False).head(10).index
    probs = counts.loc[top_pairs] / counts.loc[top_pairs].sum()
    return probs, top_pairs


def synth_od_distribution(
    df: pd.DataFrame, *, pairs: Iterable[tuple[str, str]]
) -> pd.Series:
    """Distribuição OD para os pares especificados – com proteção a div/0."""
    df_zones = assign_zone_names(df.copy())
    counts = df_zones.groupby(["PU_zone_name", "DO_zone_name"]).size()
    counts = counts.reindex(pairs, fill_value=0)

    total = counts.sum()
    if total == 0:
        # penaliza se nenhum par de interesse aparecer
        return pd.Series(0.0, index=pairs)

    return counts / total


def od_distance(
    synth_df: pd.DataFrame, *, real_probs: pd.Series, pairs: Iterable[tuple[str, str]]
) -> float:
    """Métrica: maior diferença absoluta entre distribuições OD."""
    synth_probs = synth_od_distribution(synth_df, pairs=pairs)
    return (synth_probs - real_probs).abs().max()


# --------------------------------------------------------------------------- #
#                       SUGESTÃO DE HIPERPARÂMETROS                           #
# --------------------------------------------------------------------------- #


def _suggest_hidden_dims(
    trial: optuna.trial.Trial, n_layers: int, input_dim: int
) -> List[int]:
    """Sugere unidades por camada garantindo funil decrescente.

    Caso não existam mais tamanhos menores disponíveis, o loop é encerrado
    antecipadamente; isto impede que o Optuna receba uma lista vazia.
    """
    possible_units = [512, 256, 128, 64, 32, 16, 8, 4, 2]
    hidden: List[int] = []
    units_prev = input_dim

    for i in range(n_layers):
        candidates = [u for u in possible_units if u < units_prev]
        if not candidates:
            break
        units = trial.suggest_categorical(f"units_l{i+1}", candidates)
        hidden.append(units)
        units_prev = units

    if not hidden:  # fallback para garantir que exista ao menos uma camada
        hidden = [max(2, input_dim // 2)]

    return hidden


# --------------------------------------------------------------------------- #
#                              OPTUNA OBJECTIVE                               #
# --------------------------------------------------------------------------- #


def objective(
    trial: optuna.trial.Trial,
    gmm_train: pd.DataFrame,
    X_train: np.ndarray,
    input_dim: int,
    device: str,
    scaler: StandardScaler,
    real_probs: pd.Series,
    od_pairs: Iterable[tuple[str, str]],
) -> float:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden = _suggest_hidden_dims(trial, n_layers, input_dim)

    # Latent ≤ input_dim
    max_latent = min(input_dim, 32)
    latent_dim = trial.suggest_int("latent_dim", 2, max_latent)

    epochs = trial.suggest_int("epochs", 50, 200, step=50)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

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

    # Amostragem sintética
    n_amostras = len(gmm_train)
    synth_df = amostrar_do_vae(
        model,
        n_amostras=n_amostras,
        device=device,
        scaler=scaler,
    )

    return od_distance(synth_df, real_probs=real_probs, pairs=od_pairs)


# --------------------------------------------------------------------------- #
#                           PIPELINE PRINCIPAL                                #
# --------------------------------------------------------------------------- #


def process_data():
    dados_reais_orig, *_ , GMM_FEATURES, _ = load_real_data()

    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, _ = split_dataset_weekly(
        gmm_full,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )

    scaler = StandardScaler().fit(gmm_train[LOCATION_COLS])

    X_train_df = gmm_train[["sin_hr", "cos_hr"] + LOCATION_COLS].copy()
    X_train_df[LOCATION_COLS] = scaler.transform(X_train_df[LOCATION_COLS])
    X_train = X_train_df.to_numpy(np.float32)

    # Distribuição OD real (usa apenas gmm_train)
    real_probs, od_pairs = real_od_distribution(gmm_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda tr: objective(
            tr,
            gmm_train,
            X_train,
            X_train.shape[1],
            device,
            scaler,
            real_probs,
            od_pairs,
        ),
        n_trials=30,
        show_progress_bar=True,
    )

    best = study.best_trial.params

    # Reconstrói hidden_dims apenas com as chaves existentes
    hidden_dims = tuple(
        best[k] for k in ("units_l1", "units_l2", "units_l3") if k in best
    )

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
    return vae_final, gmm_train, scaler, real_probs, od_pairs


def main():
    vae, gmm_train, scaler, real_probs, od_pairs = process_data()

    n_synth = int(len(gmm_train) * SYNTHETIC_MULTIPLIER)
    synth_df = amostrar_do_vae(
        vae,
        n_amostras=n_synth,
        device="cpu",
        batch_size=100_000,
        scaler=scaler,
    )

    metric = od_distance(synth_df, real_probs=real_probs, pairs=od_pairs)
    print("Métrica OD final:", metric)

    out_path = Path("/home-ext/caioloss/Dados/viagens_synth_vae_od.parquet")
    synth_df.to_parquet(out_path, index=False)
    print(f"Dados sintéticos salvos em {out_path}")


if __name__ == "__main__":
    log_path = Path(__file__).with_name("train_vae_od_log.txt")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with log_path.open("w") as f:
        tee_out = Tee(orig_stdout, f)
        tee_err = Tee(orig_stdout, f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            main()
