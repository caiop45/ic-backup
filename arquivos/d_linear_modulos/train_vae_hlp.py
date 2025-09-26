"""Treino de VAE+MDN focado em preservar a distribuição OD."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import contextlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from train_vae import Tee

from od_discretizer import prepare_discretized_data
from plot_od_analysis import generate_all_plots
from vae_mdn import build_latent_bank, sample_vae_mdn, train_vae_mdn
from utils.helpers import decode_hour_from_sincos


BASE_MODEL_PARAMS = {
    "embed_dim": 32,
    "hidden_dims": (8,),
    "latent_dim": 45,
    "epochs": 20,
    "batch_size": 256,
    "lr": 0.00016067507015721257,
    "kl_beta": 0.0006768259726920823,
    "mdn_weight": 1.6435551977650984,
}


# --------------------------------------------------------------------------- #
#                          MÉTRICAS DE ORIGEM-DESTINO                         #
# --------------------------------------------------------------------------- #


def real_od_distribution(df: pd.DataFrame, *, od_col: str) -> tuple[pd.Series, pd.Index]:
    counts = df[od_col].value_counts()
    top_pairs = counts.head(10).index
    probs = counts.loc[top_pairs] / counts.loc[top_pairs].sum()
    return probs, top_pairs


def synth_od_distribution(
    df: pd.DataFrame,
    *,
    pairs: Iterable[str],
    od_col: str,
) -> pd.Series:
    counts = df[od_col].value_counts()
    counts = counts.reindex(pairs, fill_value=0)
    total = counts.sum()
    if total == 0:
        return pd.Series(0.0, index=pairs)
    return counts / total


def od_distance(
    synth_df: pd.DataFrame,
    *,
    real_probs: pd.Series,
    pairs: Iterable[str],
    od_col: str,
) -> float:
    synth_probs = synth_od_distribution(synth_df, pairs=pairs, od_col=od_col)
    return (synth_probs - real_probs).abs().max()


# --------------------------------------------------------------------------- #
#                           PIPELINE PRINCIPAL                                #
# --------------------------------------------------------------------------- #


def process_data(
    mdn_components: int = 8,
    *,
    kl_beta: float | None = None,
    mdn_weight: float | None = None,
):
    print("[Pipeline] Carregando dados reais...")
    full_df, *_ , feature_columns, _ = load_real_data()

    required_cols = ["tpep_pickup_datetime", *feature_columns]
    base_df = full_df[required_cols].dropna().reset_index(drop=True)

    train_raw, val_raw, hold_raw = split_dataset_weekly(
        base_df,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    print(
        f"[Pipeline] train={len(train_raw)} | val={len(val_raw)} | hold={len(hold_raw)} | "
        f"features={feature_columns}"
    )

    train_discrete, od_artifacts = prepare_discretized_data(train_raw)
    val_discrete, _ = prepare_discretized_data(val_raw)
    hold_discrete, _ = prepare_discretized_data(hold_raw)

    for df in (train_discrete, val_discrete, hold_discrete):
        if not df.empty:
            df["hora_do_dia"] = decode_hour_from_sincos(df["sin_hr"], df["cos_hr"])
    print(
        f"[Discretização] linhas válidas (treino)={len(train_discrete)} | "
        f"pickup={len(od_artifacts.pickup_categories)} | "
        f"dropoff={len(od_artifacts.dropoff_categories)} | "
        f"od_pairs={len(od_artifacts.od_categories)}"
    )

    real_probs, od_pairs = real_od_distribution(train_discrete, od_col="od_pair")
    od_counts = train_discrete["od_id"].value_counts().sort_index()
    od_distribution = (od_counts / od_counts.sum()).to_numpy(np.float64)
    print(
        f"[Discretização] Top-5 pares OD: {list(od_pairs[:5])}"
    )
    print(
        "[Discretização] Probabilidades reais (top-5):",
        real_probs.head().round(4).to_dict(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Treinando no dispositivo: {device}")

    params = BASE_MODEL_PARAMS.copy()
    if kl_beta is not None:
        params["kl_beta"] = kl_beta
    if mdn_weight is not None:
        params["mdn_weight"] = mdn_weight
    print("[Pipeline] Usando hiperparâmetros (Trial 6 base com ajustes):", params)

    final_model = train_vae_mdn(
        train_discrete,
        pickup_cardinality=len(od_artifacts.pickup_categories),
        dropoff_cardinality=len(od_artifacts.dropoff_categories),
        hidden_dims=params["hidden_dims"],
        latent_dim=params["latent_dim"],
        embed_dim=params["embed_dim"],
        mdn_components=mdn_components,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        lr=params["lr"],
        device=device,
        kl_beta=params["kl_beta"],
        mdn_weight=params["mdn_weight"],
    )

    # Teste sem latent_bank para avaliar capacidade generativa pura do VAE
    # latent_bank = build_latent_bank(
    #     final_model,
    #     train_discrete,
    #     device=device,
    # )
    latent_bank = None

    return (
        final_model,
        train_discrete,
        val_discrete,
        hold_discrete,
        od_artifacts,
        od_distribution,
        real_probs,
        od_pairs,
        latent_bank,
        params,
    )


def plot_mdn_dispersion(
    model: torch.nn.Module,
    *,
    od_artifacts,
    od_pairs: Sequence[str],
    reference_df: pd.DataFrame,
    out_path: Path,
    device: torch.device | str = "cpu",
    latent_samples: int = 512,
) -> None:
    """Visualiza pesos e centros dos componentes MDN para pares OD selecionados."""

    od_pairs = list(od_pairs)
    if not od_pairs:
        return

    model = model.to(device).eval()

    cols = min(3, len(od_pairs))
    rows = math.ceil(len(od_pairs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()
    scatter_handles = []

    for ax, pair in zip(axes, od_pairs):
        od_id = od_artifacts.od_to_id.get(pair)
        if od_id is None:
            ax.text(0.5, 0.5, "Par OD não encontrado", ha="center", va="center")
            ax.axis("off")
            continue

        pickup_id = od_artifacts.pickup_id_for_od(od_id)
        dropoff_id = od_artifacts.dropoff_id_for_od(od_id)

        z = torch.randn(latent_samples, model.latent_dim, device=device)
        pickup = torch.full((latent_samples,), pickup_id, dtype=torch.long, device=device)
        dropoff = torch.full((latent_samples,), dropoff_id, dtype=torch.long, device=device)

        with torch.no_grad():
            pi_logits, mdn_mu, mdn_log_sigma = model.decode(z, pickup, dropoff)
        pi = F.softmax(pi_logits, dim=-1).cpu().numpy().reshape(-1)
        mu = mdn_mu.cpu().numpy().reshape(-1, 2)
        sigma = F.softplus(mdn_log_sigma).cpu().numpy().reshape(-1, 2)

        norm = np.linalg.norm(mu, axis=1, keepdims=True)
        norm[norm == 0] = 1
        unit = mu / norm
        hours = np.mod(np.arctan2(unit[:, 0], unit[:, 1]), 2 * np.pi) * 24 / (2 * np.pi)

        size_scale = sigma.mean(axis=1)
        positive = size_scale > 0
        if positive.any():
            min_positive = size_scale[positive].min()
            size_scale[~positive] = min_positive
            denom = size_scale.max()
            size_norm = size_scale / denom if denom > 0 else size_scale
        else:
            size_scale[:] = 1.0
            size_norm = size_scale
        sizes = np.clip(size_norm * 400, 20, 400)

        if pi.size:
            sc = ax.scatter(
                hours,
                pi,
                s=sizes,
                alpha=0.35,
                c=hours,
                cmap="viridis",
                edgecolors="none",
            )
            scatter_handles.append(sc)

        mask = reference_df["od_pair"] == pair
        if mask.any() and "hora_do_dia" in reference_df.columns:
            real_counts = (
                reference_df.loc[mask, "hora_do_dia"]
                .value_counts(normalize=True)
                .sort_index()
            )
            ax.step(real_counts.index, real_counts.values, where="mid", color="black", linewidth=1.0, label="Real")

        ax.set_title(pair, fontsize=9)
        ax.set_xlim(0, 24)
        ymax = pi.max() * 1.1 if pi.size else 0.01
        ax.set_ylim(0, max(0.01, ymax))
        ax.set_xlabel("Hora (μ componente)")
        ax.set_ylabel("Peso do componente")
        if mask.any():
            ax.legend(loc="upper right", fontsize=7)

    for ax in axes[len(od_pairs):]:
        ax.axis("off")

    if scatter_handles:
        fig.colorbar(
            scatter_handles[0],
            ax=axes[: len(od_pairs)],
            orientation="horizontal",
            fraction=0.05,
            pad=0.08,
            label="Hora (μ dos componentes)",
        )

    fig.suptitle("Dispersão dos componentes MDN por par OD", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    base_params = BASE_MODEL_PARAMS
    experiments = [
        ("base", base_params["kl_beta"], base_params["mdn_weight"]),
        ("klx15_mdn0.8", base_params["kl_beta"] * 15, 0.8),
        ("klx30_mdn0.6", base_params["kl_beta"] * 30, 0.6),
    ]

    plots_root = Path("graficos/od_analysis")
    plots_root.mkdir(parents=True, exist_ok=True)

    for exp_name, kl_beta, mdn_weight in experiments:
        print(f"\n[Experimento] Iniciando '{exp_name}' (kl_beta={kl_beta}, mdn_weight={mdn_weight})")
        (
            model,
            train_df,
            val_df,
            hold_df,
            od_artifacts,
            od_distribution,
            real_probs_train,
            od_pairs_train,
            latent_bank,
            used_params,
        ) = process_data(mdn_components=8, kl_beta=kl_beta, mdn_weight=mdn_weight)
        print(f"[Experimento] Hiperparâmetros efetivos: {used_params}")

        n_synth = int(len(train_df) * SYNTHETIC_MULTIPLIER)
        if n_synth <= 0:
            raise ValueError("SYNTHETIC_MULTIPLIER resultou em número de amostras <= 0")
        print(f"[Geração][{exp_name}] Amostrando {n_synth} linhas sintéticas...")

        synth_df = sample_vae_mdn(
            model,
            n_samples=n_synth,
            od_artifacts=od_artifacts,
            od_distribution=od_distribution,
            device="cpu",
            latent_bank=latent_bank,
        )
        del latent_bank
        print(f"[Geração][{exp_name}] Amostras geradas (head):")
        print(synth_df.head())

        train_metric = od_distance(
            synth_df,
            real_probs=real_probs_train,
            pairs=od_pairs_train,
            od_col="od_pair",
        )
        print(f"[Geração][{exp_name}] Métrica OD (treino) max|Δ| = {train_metric:.4f}")

        val_probs, val_pairs = real_od_distribution(val_df, od_col="od_pair")
        val_metric = od_distance(
            synth_df,
            real_probs=val_probs,
            pairs=val_pairs,
            od_col="od_pair",
        )
        print(f"[Geração][{exp_name}] Métrica OD (validação) max|Δ| = {val_metric:.4f}")

        if not hold_df.empty:
            hold_probs, hold_pairs = real_od_distribution(hold_df, od_col="od_pair")
            hold_metric = od_distance(
                synth_df,
                real_probs=hold_probs,
                pairs=hold_pairs,
                od_col="od_pair",
            )
            print(f"[Geração][{exp_name}] Métrica OD (hold-out) max|Δ| = {hold_metric:.4f}")

        out_path = Path(f"/home-ext/caioloss/Dados/viagens_synth_vae_od_{exp_name}.parquet")
        synth_df.to_parquet(out_path, index=False)
        print(f"[Geração][{exp_name}] Dados sintéticos salvos em {out_path}")

        val_plots = plots_root / "validation"
        print(
            f"[Gráficos][{exp_name}] Gerando comparativos (validação) em {val_plots.resolve()}..."
        )
        scale = len(val_df) / max(len(synth_df), 1)
        generate_all_plots(
            val_df,
            synth_df,
            val_plots,
            tag=exp_name,
            synth_scale=scale,
        )

        top_pairs_dispersion = val_df["od_pair"].value_counts().head(6).index.tolist()
        dispersion_path = val_plots / f"mdn_dispersion_{exp_name}.png"
        print(f"[Gráficos][{exp_name}] Plotando dispersão MDN em {dispersion_path.resolve()}...")
        plot_mdn_dispersion(
            model,
            od_artifacts=od_artifacts,
            od_pairs=top_pairs_dispersion,
            reference_df=val_df,
            out_path=dispersion_path,
            device="cpu",
        )

        if not hold_df.empty:
            hold_plots = plots_root / "holdout"
            print(
                f"[Gráficos][{exp_name}] Gerando comparativos (hold-out) em {hold_plots.resolve()}..."
            )
            generate_all_plots(
                hold_df,
                synth_df,
                hold_plots,
                tag=exp_name,
                synth_scale=len(hold_df) / max(len(synth_df), 1),
            )

    print("[Gráficos] Arquivos gerados:")
    for path in sorted(plots_root.rglob("*.png")):
        print(f"   - {path}")


if __name__ == "__main__":
    log_path = Path(__file__).with_name("train_vae_od_log.txt")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with log_path.open("w") as f:
        tee_out = Tee(orig_stdout, f)
        tee_err = Tee(orig_stdout, f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            main()
