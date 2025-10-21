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
from torch.utils.data import DataLoader, TensorDataset

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from train_vae import Tee

from od_discretizer import prepare_discretized_data, apply_artifacts
from plot_od_analysis import generate_all_plots
from vae_mdn import build_latent_bank, sample_vae_mdn, train_vae_mdn
from utils.helpers import decode_hour_from_sincos

SCRIPT_DIR = Path(__file__).resolve().parent


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
#                     AGREGAÇÃO OD × HORA × DIA (CSV)                         #
# --------------------------------------------------------------------------- #

def _aggregate_od_hour_day(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por OD × hora_do_dia × dia_da_semana e conta n_viagens.

    Espera colunas: 'tpep_pickup_datetime' (datetime64) e 'od_pair'.
    Retorna DataFrame com colunas: 'od', 'hora_do_dia', 'dia_da_semana', 'n_viagens'.
    """
    if df.empty:
        return pd.DataFrame(columns=["od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio

    tmp = df[["tpep_pickup_datetime", "od_pair"]].copy()
    # Garante dtype datetime
    tmp["tpep_pickup_datetime"] = pd.to_datetime(tmp["tpep_pickup_datetime"], errors="coerce")
    tmp = tmp.dropna(subset=["tpep_pickup_datetime", "od_pair"])  # remove linhas inválidas

    # Extrai hora (0-23) e dia da semana (0=Seg .. 6=Dom)
    tmp["hora_do_dia"] = tmp["tpep_pickup_datetime"].dt.hour.astype(int)
    tmp["dia_da_semana"] = tmp["tpep_pickup_datetime"].dt.dayofweek.astype(int)
    tmp["od"] = tmp["od_pair"].astype(str)

    agg = (
        tmp.groupby(["od", "hora_do_dia", "dia_da_semana"]).size().reset_index(name="n_viagens")
    )
    # Ordenação estável ajuda leitura humana
    agg = agg.sort_values(["od", "dia_da_semana", "hora_do_dia"]).reset_index(drop=True)
    return agg


def _aggregate_od_hour_day_synth(df: pd.DataFrame, *, default_dow: int = 0) -> pd.DataFrame:
    """Agrega sintético por OD × hora × dia.

    Para dados sintéticos não há datetime, então o dia_da_semana é fixo em
    `default_dow` (padrão 0=segunda) apenas para manter o esquema idêntico.
    A hora é obtida de 'hora_target' (quando existir) ou de 'hora_do_dia',
    caindo para decode a partir de 'sin_hr'/'cos_hr' se necessário.
    """
    if df.empty:
        return pd.DataFrame(columns=["od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio

    tmp = df.copy()
    # Garante coluna de OD como string
    tmp["od"] = tmp["od_pair"].astype(str)

    # Deriva hora (inteiro 0..23)
    if "hora_target" in tmp.columns:
        hours = tmp["hora_target"].astype(int)
    elif "hora_do_dia" in tmp.columns:
        hours = np.floor(tmp["hora_do_dia"].astype(float)).astype(int)
    elif {"sin_hr", "cos_hr"}.issubset(tmp.columns):
        hours = np.floor(decode_hour_from_sincos(tmp["sin_hr"], tmp["cos_hr"]).astype(float)).astype(int)
    else:
        # Sem informação de hora; retorna vazio coerente
        return pd.DataFrame(columns=["od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio
    hours = hours.clip(0, 23)

    tmp["hora_do_dia"] = hours
    tmp["dia_da_semana"] = int(default_dow)

    agg = (
        tmp.groupby(["od", "hora_do_dia", "dia_da_semana"]).size().reset_index(name="n_viagens")
    )
    agg = agg.sort_values(["od", "dia_da_semana", "hora_do_dia"]).reset_index(drop=True)
    return agg


# --------------------------------------------------------------------------- #
#                               DEBUG UTILITIES                               #
# --------------------------------------------------------------------------- #


def _normalize_histogram(values: np.ndarray) -> np.ndarray:
    s = float(values.sum())
    if s <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values.astype(np.float64) / s


def _safe_jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_histogram(p)
    q = _normalize_histogram(q)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
    kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


def _chi2_stat(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalize_histogram(p)
    q = _normalize_histogram(q)
    denom = q + eps
    return float(np.sum(((p - q) ** 2) / denom))


def _emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_histogram(p)
    q = _normalize_histogram(q)
    cdf_diff = np.cumsum(p) - np.cumsum(q)
    return float(np.sum(np.abs(cdf_diff)))


def _print_hist_24(tag: str, hist: np.ndarray) -> None:
    rounded = [float(f"{x:.4f}") for x in hist.tolist()]
    peak = int(np.argmax(hist)) if hist.size else -1
    print(f"[Debug][{tag}] hist24={rounded} | peak={peak} | sum={float(hist.sum()):.6f} | std={float(hist.std()):.6f}")


def _validate_sincos(df: pd.DataFrame, *, label: str) -> None:
    if df.empty or not {"sin_hr", "cos_hr"}.issubset(df.columns):
        print(f"[Debug][{label}] sin/cos ausentes ou DF vazio; pulando validação de sincos.")
        return
    sin_vals = df["sin_hr"].to_numpy(np.float64)
    cos_vals = df["cos_hr"].to_numpy(np.float64)
    norms = np.sqrt(sin_vals**2 + cos_vals**2)
    within = np.logical_and(norms >= 0.85, norms <= 1.15).mean()
    print(
        f"[Debug][{label}] sincos: mean_norm={float(norms.mean()):.4f} std_norm={float(norms.std()):.4f} "
        f"min_norm={float(norms.min()):.4f} max_norm={float(norms.max()):.4f} frac_in_[0.85,1.15]={within:.3f}"
    )


def _print_dataset_info(label: str, df: pd.DataFrame) -> None:
    print(f"[Debug][DF][{label}] rows={len(df)} | cols={list(df.columns)}")
    if "tpep_pickup_datetime" in df.columns and not df.empty:
        try:
            dt_min = pd.to_datetime(df["tpep_pickup_datetime"]).min()
            dt_max = pd.to_datetime(df["tpep_pickup_datetime"]).max()
            print(f"[Debug][DF][{label}] datetime range: {dt_min} .. {dt_max}")
        except Exception as e:
            print(f"[Debug][DF][{label}] falha ao parsear datetimes: {e}")
    for col in ["sin_hr", "cos_hr", "pickup_id", "dropoff_id", "od_id", "od_pair", "hora_do_dia"]:
        if col in df.columns:
            null_frac = float(df[col].isna().mean())
            print(f"[Debug][DF][{label}] null_frac[{col}]={null_frac:.4f}")


def _print_hour_stats(label: str, df: pd.DataFrame) -> None:
    hist = _hour_histogram(df)
    _print_hist_24(f"{label}:hour", hist)
    top3 = np.argsort(hist)[::-1][:3].tolist()
    print(f"[Debug][{label}] top3_hours={top3} top3_vals={[float(f'{hist[i]:.4f}') for i in top3]}")


def _print_hour_compare(tag: str, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    r = _hour_histogram(real_df)
    s = _hour_histogram(synth_df)
    l1 = float(np.abs(s - r).sum())
    l2 = float(np.sqrt(np.square(s - r).sum()))
    jsd = _safe_jsd(r, s)
    chi2 = _chi2_stat(r, s)
    emd = _emd_1d(r, s)
    print(
        f"[HoraCmp][{tag}] L1={l1:.4f} L2={l2:.4f} JSD={jsd:.4f} chi2={chi2:.4f} EMD={emd:.4f} | "
        f"std_real={float(r.std()):.4f} std_synth={float(s.std()):.4f}"
    )


def _print_od_coverage(tag: str, real_df: pd.DataFrame, synth_df: pd.DataFrame, top_k: int = 10) -> None:
    if real_df.empty or synth_df.empty or "od_pair" not in real_df.columns or "od_pair" not in synth_df.columns:
        print(f"[ODcov][{tag}] dados insuficientes para cobertura OD.")
        return
    top_pairs = real_df["od_pair"].value_counts().head(top_k).index
    present = synth_df["od_pair"].value_counts().reindex(top_pairs, fill_value=0)
    coverage = (present > 0).mean()
    print(f"[ODcov][{tag}] top_k={top_k} coverage={coverage:.3f} counts_synth={present.to_dict()}")


# --------------------------------------------------------------------------- #
#                           PIPELINE PRINCIPAL                                #
# --------------------------------------------------------------------------- #


def process_data(
    mdn_components: int = 8,
    *,
    kl_beta: float | None = None,
    mdn_weight: float | None = None,
    train_overrides: dict | None = None,
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
    _print_dataset_info("train_raw", train_raw)
    _print_dataset_info("val_raw", val_raw)
    _print_dataset_info("hold_raw", hold_raw)

    # Discretiza treino e REAPROVEITA o vocabulário para val/hold
    train_discrete, od_artifacts = prepare_discretized_data(train_raw)
    from utils.zone_id import assign_zone_names
    val_enriched = assign_zone_names(val_raw.copy())
    hold_enriched = assign_zone_names(hold_raw.copy()) if not hold_raw.empty else hold_raw.copy()
    val_discrete = apply_artifacts(val_enriched, od_artifacts)
    hold_discrete = apply_artifacts(hold_enriched, od_artifacts) if not hold_enriched.empty else hold_enriched

    for df in (train_discrete, val_discrete, hold_discrete):
        if not df.empty:
            df["hora_do_dia"] = decode_hour_from_sincos(df["sin_hr"], df["cos_hr"])
    print(
        f"[Discretização] linhas válidas (treino)={len(train_discrete)} | "
        f"pickup={len(od_artifacts.pickup_categories)} | "
        f"dropoff={len(od_artifacts.dropoff_categories)} | "
        f"od_pairs={len(od_artifacts.od_categories)}"
    )
    _validate_sincos(train_discrete, label="train_discrete")
    _validate_sincos(val_discrete, label="val_discrete")
    _validate_sincos(hold_discrete, label="hold_discrete")
    _print_hour_stats("train_discrete", train_discrete)
    _print_hour_stats("val_discrete", val_discrete)
    _print_hour_stats("hold_discrete", hold_discrete)

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
    if not od_counts.empty:
        print(f"[Discretização] od_counts: min={int(od_counts.min())} max={int(od_counts.max())} "
              f"mean={float(od_counts.mean()):.2f} std={float(od_counts.std()):.2f} unique_od={od_counts.size}")
    print(f"[Discretização] od_distribution: sum={float(od_distribution.sum()):.6f} "
          f"min={float(od_distribution.min()):.6f} max={float(od_distribution.max()):.6f} "
          f"std={float(od_distribution.std()):.6f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Treinando no dispositivo: {device}")

    params = BASE_MODEL_PARAMS.copy()
    if kl_beta is not None:
        params["kl_beta"] = kl_beta
    if mdn_weight is not None:
        params["mdn_weight"] = mdn_weight
    print("[Pipeline] Usando hiperparâmetros (Trial 6 base com ajustes):", params)

    extra = train_overrides or {}
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
        **extra,
    )

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


def _hour_histogram(df: pd.DataFrame, *, hour_col: str = "hora_do_dia") -> np.ndarray:
    hist = np.zeros(24, dtype=np.float64)
    if df.empty or hour_col not in df.columns:
        return hist
    hours = df[hour_col].to_numpy()
    bins = np.clip(np.floor(hours).astype(int), 0, 23)
    for b in bins:
        hist[b] += 1
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist


def _print_hour_metrics(tag: str, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> None:
    real_h = _hour_histogram(real_df)
    synth_h = _hour_histogram(synth_df)
    diff = synth_h - real_h
    peak_real = int(np.argmax(real_h))
    peak_synth = int(np.argmax(synth_h))
    l1 = np.abs(diff).sum()
    madrugada_r, madrugada_s = real_h[0:4].sum(), synth_h[0:4].sum()
    tarde_r, tarde_s = real_h[15:19].sum(), synth_h[15:19].sum()
    noite_r, noite_s = real_h[20:24].sum(), synth_h[20:24].sum()
    print(
        f"[Hora][{tag}] pico_real={peak_real} | pico_synth={peak_synth} | L1={l1:.3f}"
    )
    print(
        f"[Hora][{tag}] madrugada(0-3) real={madrugada_r:.3f} synth={madrugada_s:.3f} | "
        f"tarde(15-18) real={tarde_r:.3f} synth={tarde_s:.3f} | "
        f"noite(20-23) real={noite_r:.3f} synth={noite_s:.3f}"
    )


def _print_top_od_hour_metrics(tag: str, real_df: pd.DataFrame, synth_df: pd.DataFrame, top_k: int = 6) -> None:
    if real_df.empty or synth_df.empty:
        return
    top_pairs = real_df["od_pair"].value_counts().head(top_k).index.tolist()
    print(f"[HoraOD][{tag}] Avaliando top-{top_k} pares: {top_pairs}")
    for pair in top_pairs:
        r = real_df[real_df["od_pair"] == pair]
        s = synth_df[synth_df["od_pair"] == pair]
        real_h = _hour_histogram(r)
        synth_h = _hour_histogram(s)
        diff = synth_h - real_h
        l1 = np.abs(diff).sum()
        tarde_gap = (synth_h[15:19].sum() - real_h[15:19].sum())
        noite_gap = (synth_h[20:24].sum() - real_h[20:24].sum())
        print(
            f"  - {pair}: L1={l1:.3f} | Δtarde(15-18)={tarde_gap:+.3f} | Δnoite(20-23)={noite_gap:+.3f}"
        )


def diagnose_encoder(
    model: torch.nn.Module,
    df: pd.DataFrame,
    *,
    label: str,
    device: torch.device | str,
    sample_size: int = 40_000,
) -> None:
    if df.empty:
        print(f"[Diag][{label}] DataFrame vazio, pulando análise do encoder.")
        return

    subset = df.sample(min(sample_size, len(df)), random_state=42)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(subset[["sin_hr", "cos_hr"]].to_numpy(np.float32)),
            torch.tensor(subset["pickup_id"].to_numpy(np.int64)),
            torch.tensor(subset["dropoff_id"].to_numpy(np.int64)),
        ),
        batch_size=4096,
        shuffle=False,
        drop_last=False,
    )

    model = model.to(device).eval()
    mu_list: list[np.ndarray] = []
    logvar_list: list[np.ndarray] = []

    with torch.no_grad():
        for sincos, pickup_id, dropoff_id in loader:
            sincos = sincos.to(device)
            pickup_id = pickup_id.to(device)
            dropoff_id = dropoff_id.to(device)
            mu, logvar = model.encode(sincos, pickup_id, dropoff_id)
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())

    mu = np.concatenate(mu_list, axis=0)
    logvar = np.concatenate(logvar_list, axis=0)
    sigma = np.exp(0.5 * logvar)
    kl = -0.5 * (1 + logvar - mu**2 - np.exp(logvar)).sum(axis=1)

    print(
        f"[Diag][{label}] |μ| mean={np.abs(mu).mean():.4f}, median={np.median(np.abs(mu)):.4f}, "
        f"σ mean={sigma.mean():.4f}, median={np.median(sigma):.4f}, KL mean={kl.mean():.4f}, std={kl.std():.4f}"
    )


def diagnose_decoder(
    model: torch.nn.Module,
    od_artifacts,
    *,
    od_pairs: Sequence[str],
    device: torch.device | str,
    latent_samples: int = 1024,
) -> None:
    if not od_pairs:
        return

    model = model.to(device).eval()
    latent_dim = model.latent_dim
    print(f"[Diag][decoder] Avaliando {len(od_pairs)} pares com {latent_samples} amostras latentes...")

    for pair in od_pairs:
        od_id = od_artifacts.od_to_id.get(pair)
        if od_id is None:
            print(f"  - {pair}: OD não encontrado nos artefatos.")
            continue
        pickup_id = od_artifacts.pickup_id_for_od(od_id)
        dropoff_id = od_artifacts.dropoff_id_for_od(od_id)

        z = torch.randn(latent_samples, latent_dim, device=device)
        pickup = torch.full((latent_samples,), pickup_id, dtype=torch.long, device=device)
        dropoff = torch.full((latent_samples,), dropoff_id, dtype=torch.long, device=device)

        with torch.no_grad():
            pi_logits, mdn_mu, mdn_log_sigma = model.decode(z, pickup, dropoff)

        pi = F.softmax(pi_logits, dim=-1).cpu().numpy()
        mu = mdn_mu.cpu().numpy()
        sigma = F.softplus(mdn_log_sigma).cpu().numpy()

        pi_mean = pi.mean(axis=0)
        entropy = -(pi_mean * np.log(pi_mean + 1e-8)).sum()
        top_idx = np.argsort(pi_mean)[::-1][:3]

        comp_info = []
        for idx in top_idx:
            mu_vec = mu[:, idx, :]
            sigma_vec = sigma[:, idx, :]
            norm = np.linalg.norm(mu_vec, axis=1, keepdims=True)
            norm[norm == 0] = 1
            hours = (np.mod(np.arctan2(mu_vec[:, 0], mu_vec[:, 1]), 2 * np.pi) * 24 / (2 * np.pi)) % 24
            comp_info.append(
                {
                    "component": int(idx),
                    "pi": float(pi_mean[idx]),
                    "hora_média": float(hours.mean()),
                    "hora_std": float(hours.std()),
                    "sigma_mean": float(sigma_vec.mean()),
                }
            )

        active = int((pi_mean > (1.0 / len(pi_mean) * 0.3)).sum())
        print(
            f"  - {pair}: entropia={entropy:.3f}, comp_ativos≈{active}, top_componentes={comp_info}"
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

    # Execução única com valores existentes: usamos mult=1.25 e mdn_weight=1.4
    mult = 1.25
    weight = 1.4
    exp_name = f"klx{str(mult).replace('.', 'p')}_mdw{str(weight).replace('.', 'p')}"
    kl_beta = base_params["kl_beta"] * mult
    mdn_weight = weight

    plots_root = SCRIPT_DIR / "graficos" / "od_analysis"
    plots_root.mkdir(parents=True, exist_ok=True)

    diag_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"\n[Execução] Iniciando '{exp_name}' (kl_beta={kl_beta}, mdn_weight={mdn_weight})"
    )
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
    print(f"[Execução] Hiperparâmetros efetivos: {used_params}")

    # Infos adicionais dos DFs discretizados
    _print_dataset_info("train_df", train_df)
    _print_dataset_info("val_df", val_df)
    _print_dataset_info("hold_df", hold_df)
    _print_hour_stats("train_df", train_df)
    _print_hour_stats("val_df", val_df)
    _print_hour_stats("hold_df", hold_df)

    diagnose_encoder(
        model,
        train_df,
        label=f"{exp_name}-train",
        device=diag_device,
    )
    diagnose_encoder(
        model,
        val_df,
        label=f"{exp_name}-val",
        device=diag_device,
    )
    if not hold_df.empty:
        diagnose_encoder(
            model,
            hold_df,
            label=f"{exp_name}-hold",
            device=diag_device,
        )
    top_pairs_diag = val_df["od_pair"].value_counts().head(5).index.tolist()
    diagnose_decoder(
        model,
        od_artifacts,
        od_pairs=top_pairs_diag,
        device=diag_device,
    )

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
    _print_dataset_info("synth_df", synth_df)
    _validate_sincos(synth_df, label="synth_df")
    _print_hour_stats("synth_df", synth_df)

    train_metric = od_distance(
        synth_df,
        real_probs=real_probs_train,
        pairs=od_pairs_train,
        od_col="od_pair",
    )
    print(f"[Geração][{exp_name}] Métrica OD (treino) max|Δ| = {train_metric:.4f}")

    _print_hour_metrics(f"{exp_name}:train", train_df, synth_df)
    _print_hour_compare(f"{exp_name}:train", train_df, synth_df)
    _print_od_coverage(f"{exp_name}:train", train_df, synth_df, top_k=10)

    val_probs, val_pairs = real_od_distribution(val_df, od_col="od_pair")
    val_metric = od_distance(
        synth_df,
        real_probs=val_probs,
        pairs=val_pairs,
        od_col="od_pair",
    )
    print(f"[Geração][{exp_name}] Métrica OD (validação) max|Δ| = {val_metric:.4f}")
    _print_hour_metrics(f"{exp_name}:val", val_df, synth_df)
    _print_hour_compare(f"{exp_name}:val", val_df, synth_df)
    _print_top_od_hour_metrics(f"{exp_name}:val", val_df, synth_df, top_k=6)
    _print_od_coverage(f"{exp_name}:val", val_df, synth_df, top_k=10)

    if not hold_df.empty:
        hold_probs, hold_pairs = real_od_distribution(hold_df, od_col="od_pair")
        hold_metric = od_distance(
            synth_df,
            real_probs=hold_probs,
            pairs=hold_pairs,
            od_col="od_pair",
        )
        print(f"[Geração][{exp_name}] Métrica OD (hold-out) max|Δ| = {hold_metric:.4f}")
        _print_hour_metrics(f"{exp_name}:hold", hold_df, synth_df)
        _print_hour_compare(f"{exp_name}:hold", hold_df, synth_df)
        _print_top_od_hour_metrics(f"{exp_name}:hold", hold_df, synth_df, top_k=6)
        _print_od_coverage(f"{exp_name}:hold", hold_df, synth_df, top_k=10)

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
    print(
        f"[Gráficos][{exp_name}] Plotando dispersão MDN em {dispersion_path.resolve()}..."
    )
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

    # ------------------------------------------------------------------
    # CSV: viagens agregadas por OD × hora_do_dia × dia_da_semana
    # ------------------------------------------------------------------
    try:
        save_dir = SCRIPT_DIR / "save_data"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) CSV geral (treino + validação + hold-out)
        if not hold_df.empty:
            all_real_df = pd.concat([train_df, val_df, hold_df], ignore_index=True)
        else:
            all_real_df = pd.concat([train_df, val_df], ignore_index=True)

        agg_all = _aggregate_od_hour_day(all_real_df)
        out_all = save_dir / "od_hora_dia_geral.csv"
        agg_all.to_csv(out_all, index=False)
        print(f"[CSV] Agregado geral salvo em: {out_all}")

        # 2) CSV apenas validação
        agg_val = _aggregate_od_hour_day(val_df)
        out_val = save_dir / "od_hora_dia_validacao.csv"
        agg_val.to_csv(out_val, index=False)
        print(f"[CSV] Agregado validação salvo em: {out_val}")

        # 3) CSV sintético
        agg_synth = _aggregate_od_hour_day_synth(synth_df, default_dow=0)
        out_synth = save_dir / "od_hora_dia_sintetico.csv"
        agg_synth.to_csv(out_synth, index=False)
        print(f"[CSV] Agregado sintético salvo em: {out_synth}")
    except Exception as e:
        print(f"[CSV] Falha ao gerar/salvar agregados OD×hora×dia: {e}")

    print("[Gráficos] Arquivos gerados:")
    for path in sorted(plots_root.rglob("*.png")):
        print(f"   - {path}")


if __name__ == "__main__":
    log_path = SCRIPT_DIR / "logs" / "train_vae_od_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    with log_path.open("w") as f:
        tee_out = Tee(orig_stdout, f)
        tee_err = Tee(orig_stdout, f)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            main()
