"""Script de diagnósticos para VAE+MDN focado em hora do dia.

Este script treina (ou reusa o pipeline existente) e imprime métricas que
permitem isolar a origem do desvio na distribuição horária:
 - Calibração do prior p(h|OD) vs. real (train/val)
 - Varredura de temperatura no prior (τ) sem alterar o treinamento
 - Diagnóstico por par OD (top-k) do prior
 - Amostragem somente do MDN (sem prior de horas) para verificar o papel do MDN

Saída: logs em stdout, apropriados para investigação posterior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import math
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Garante que o diretório deste script esteja no sys.path para imports locais
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Importa pipeline existente
from train_vae_hlp import process_data
import os
from vae_mdn import VAEMDN, sample_vae_mdn, build_latent_bank
from od_discretizer import ODDiscretizerArtifacts
from utils.helpers import decode_hour_from_sincos, set_global_seed
from mdn_utils import sample_from_mdn


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


def _emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    p = _normalize_histogram(p)
    q = _normalize_histogram(q)
    cdf_diff = np.cumsum(p) - np.cumsum(q)
    return float(np.sum(np.abs(cdf_diff)))


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


def _print_hist_24(tag: str, hist: np.ndarray) -> None:
    rounded = [float(f"{x:.4f}") for x in hist.tolist()]
    peak = int(np.argmax(hist)) if hist.size else -1
    print(
        f"[Debug][{tag}] hist24={rounded} | peak={peak} | sum={float(hist.sum()):.6f} | std={float(hist.std()):.6f}"
    )


def _plot_hour_histogram_compare(
    *,
    real_hist: np.ndarray,
    synth_hist: np.ndarray,
    out_path: Path,
    title: str = "Distribuição por hora — Real vs Sintética",
) -> None:
    """Gera um gráfico de barras com 24 bins comparando real vs. sintético.

    O arquivo é sobrescrito a cada execução (mesmo caminho).
    """
    hours = np.arange(24)
    real = _normalize_histogram(real_hist)
    syn = _normalize_histogram(synth_hist)

    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(hours - width / 2, real, width=width, label="Real", color="#4C78A8")
    ax.bar(hours + width / 2, syn, width=width, label="Sintético", color="#F58518")
    ax.set_xticks(hours)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Proporção")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _hour_target_histogram(df: pd.DataFrame) -> np.ndarray:
    """Histograma (24 bins) das horas-alvo (hora_target) em DataFrame sintético."""
    if df.empty or "hora_target" not in df.columns:
        return np.zeros(24, dtype=np.float64)
    hours = df["hora_target"].to_numpy()
    bins = np.clip(np.floor(hours).astype(int), 0, 23)
    hist = np.zeros(24, dtype=np.float64)
    for b in bins:
        hist[b] += 1.0
    s = hist.sum()
    return (hist / s) if s > 0 else hist


def _circular_err_stats_for_targets(df_syn: pd.DataFrame, target_hours: list[int]) -> None:
    """Imprime métricas de alinhamento restritas a horas-alvo específicas."""
    if df_syn.empty:
        return
    if not {"hora_target", "hora_do_dia"}.issubset(df_syn.columns):
        return
    sel = df_syn[df_syn["hora_target"].astype(int).isin(target_hours)]
    if sel.empty:
        print(f"[Investigate] Nenhuma amostra com hora_target em {target_hours}.")
        return
    tgt = sel["hora_target"].to_numpy()
    pred = sel["hora_do_dia"].to_numpy()
    diff = _circular_hour_diff(pred, tgt)
    acc0 = float((diff == 0).mean())
    acc1 = float((diff <= 1).mean())
    mean_d = float(diff.mean())
    print(
        f"[Investigate][SubsetAlign] hours={target_hours} | acc0={acc0:.3f} acc±1={acc1:.3f} mean_circ_diff={mean_d:.3f} | n={len(sel)}"
    )
    # Estatística por hora-alvo individual
    for h in target_hours:
        row = sel[sel["hora_target"].astype(int) == h]
        if row.empty:
            continue
        d = _circular_hour_diff(row["hora_do_dia"].to_numpy(), row["hora_target"].to_numpy())
        print(f"  - hour_target={h}: acc0={(d==0).mean():.3f} acc±1={(d<=1).mean():.3f} meanΔ={d.mean():.3f} n={len(row)}")


def _confusion_for_targets(df_syn: pd.DataFrame, target_hours: list[int], topk: int = 4) -> None:
    """Mostra, para cada hora-alvo, as horas previstas mais frequentes (confusão)."""
    if df_syn.empty:
        return
    if not {"hora_target", "hora_do_dia"}.issubset(df_syn.columns):
        return
    print(f"[Investigate][Confusion] Top-{topk} previstos por hora_target em {target_hours}")
    for h in target_hours:
        sub = df_syn[df_syn["hora_target"].astype(int) == h]
        if sub.empty:
            print(f"  - hour_target={h}: sem amostras.")
            continue
        counts = sub["hora_do_dia"].astype(int).value_counts(normalize=True).sort_values(ascending=False)
        top = counts.head(topk)
        print(f"  - hour_target={h}: {top.to_dict()}")


@torch.no_grad()
def _subset_od_differences(
    tag: str,
    model: VAEMDN,
    od_artifacts: ODDiscretizerArtifacts,
    val_df: pd.DataFrame,
    df_syn: pd.DataFrame,
    *,
    subset_hours: list[int],
    top_k: int = 15,
    device: torch.device | str = "cpu",
) -> None:
    """Lista top-K pares OD com maior Δ_sub = real_sub - synth_sub nas horas de interesse.

    Também reporta prior_sub obtido diretamente de hour_logits(pu,do) do modelo, para
    isolar se a falta é do prior ou do decoder.
    """
    if val_df.empty or df_syn.empty:
        return
    subset_hours = [int(h) % 24 for h in subset_hours]

    # Massa por OD no subconjunto (real e synth)
    def _mass_by_od(df: pd.DataFrame) -> pd.Series:
        if df.empty or "od_pair" not in df.columns or "hora_do_dia" not in df.columns:
            return pd.Series(dtype=float)
        sub = df[df["hora_do_dia"].astype(int).isin(subset_hours)]
        counts = sub["od_pair"].value_counts()
        total = counts.sum()
        return counts / total if total > 0 else counts

    real_mass = _mass_by_od(val_df)
    synth_mass = _mass_by_od(df_syn)
    # Alinha índices e calcula delta
    all_pairs = real_mass.index.union(synth_mass.index)
    real_mass = real_mass.reindex(all_pairs, fill_value=0.0)
    synth_mass = synth_mass.reindex(all_pairs, fill_value=0.0)
    delta = (real_mass - synth_mass).sort_values(ascending=False)

    # Seleciona top-K onde real > synth
    top_pairs = delta.head(top_k)
    if top_pairs.empty:
        print(f"[Investigate][SubsetOD][{tag}] Sem pares OD com delta positivo nas horas {subset_hours}.")
        return

    print(f"[Investigate][SubsetOD][{tag}] top_k={top_k} horas={subset_hours}")

    # Prepara uma chamada por par ao prior p(h|OD)
    model = model.to(device).eval()
    for pair, dval in top_pairs.items():
        od_id = od_artifacts.od_to_id.get(pair)
        if od_id is None:
            continue
        pu = od_artifacts.pickup_id_for_od(od_id)
        do = od_artifacts.dropoff_id_for_od(od_id)
        pu_t = torch.tensor([pu], dtype=torch.long, device=device)
        do_t = torch.tensor([do], dtype=torch.long, device=device)
        logits = model.hour_logits(pu_t, do_t)  # [1,24]
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
        prior_sub = float(probs[subset_hours].sum())  # p(hours∈subset | OD)

        # Shares dentro do subconjunto (across-OD normalizados no subset)
        r_share = float(real_mass.get(pair, 0.0))
        s_share = float(synth_mass.get(pair, 0.0))

        # Taxas condicionais por OD: p(hours∈subset | OD) em real e synth
        val_pair = val_df[val_df["od_pair"] == pair]
        synth_pair = df_syn[df_syn["od_pair"] == pair]
        real_total = len(val_pair)
        real_sub_cnt = int((val_pair["hora_do_dia"].astype(int).isin(subset_hours)).sum()) if real_total > 0 else 0
        real_sub_rate = (real_sub_cnt / real_total) if real_total > 0 else float("nan")
        synth_total = len(synth_pair)
        synth_sub_cnt = int((synth_pair["hora_do_dia"].astype(int).isin(subset_hours)).sum()) if synth_total > 0 else 0
        synth_sub_rate = (synth_sub_cnt / synth_total) if synth_total > 0 else float("nan")
        od_weight_real = real_total / max(len(val_df), 1)

        print(
            f"  - {pair}: share_sub(real)={r_share:.4f} share_sub(synth)={s_share:.4f} Δshare={r_share - s_share:+.4f} | "
            f"p(sub|OD): real={real_sub_rate:.3f} prior={prior_sub:.3f} synth={synth_sub_rate:.3f} | weight_real={od_weight_real:.4f}"
        )


@torch.no_grad()
def _aggregate_prior_hist(
    model: VAEMDN,
    df: pd.DataFrame,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 8192,
    temperature: float | None = None,
) -> np.ndarray:
    """Agrega a distribuição predita do prior p(h|OD) sobre um DataFrame.

    Retorna histograma de 24 bins (probabilidades).
    """
    model = model.to(device).eval()
    hist = np.zeros(24, dtype=np.float64)

    if df.empty:
        return hist

    # Constrói batches
    n = len(df)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        pickup = torch.tensor(df["pickup_id"].iloc[start:end].to_numpy(np.int64), device=device)
        dropoff = torch.tensor(df["dropoff_id"].iloc[start:end].to_numpy(np.int64), device=device)
        logits = model.hour_logits(pickup, dropoff)
        if temperature is not None and temperature > 0:
            logits = logits / float(temperature)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        hist += probs.sum(axis=0)

    return _normalize_histogram(hist)


def _compare_hists(tag: str, real: np.ndarray, pred: np.ndarray) -> None:
    real = _normalize_histogram(real)
    pred = _normalize_histogram(pred)
    l1 = float(np.abs(pred - real).sum())
    l2 = float(np.sqrt(np.square(pred - real).sum()))
    jsd = _safe_jsd(real, pred)
    emd = _emd_1d(real, pred)
    print(
        f"[Compare][{tag}] L1={l1:.4f} L2={l2:.4f} JSD={jsd:.4f} EMD={emd:.4f} | "
        f"peak_real={int(np.argmax(real))} peak_pred={int(np.argmax(pred))}"
    )
    _print_hist_24(f"{tag}:real", real)
    _print_hist_24(f"{tag}:pred", pred)


def _circular_hour_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Diferença circular mínima em horas entre vetores inteiros [0,23]."""
    a = a.astype(np.int64) % 24
    b = b.astype(np.int64) % 24
    d = np.abs(a - b)
    return np.minimum(d, 24 - d)


def _hour_alignment_metrics(df_syn: pd.DataFrame) -> tuple[float, float, float]:
    """Retorna (acc0, acc_pm1, mean_abs_circ_diff) entre hora_do_dia e hora_target."""
    if df_syn.empty or "hora_target" not in df_syn.columns or "hora_do_dia" not in df_syn.columns:
        return (float("nan"), float("nan"), float("nan"))
    tgt = df_syn["hora_target"].to_numpy()
    pred = df_syn["hora_do_dia"].to_numpy()
    diff = _circular_hour_diff(pred, tgt)
    acc0 = float((diff == 0).mean())
    acc_pm1 = float((diff <= 1).mean())
    mean_diff = float(diff.mean())
    return acc0, acc_pm1, mean_diff


def _peak_hour(hist: np.ndarray) -> int:
    return int(np.argmax(hist)) if hist.size else -1


def _top_od_hour_metrics(tag: str, real_df: pd.DataFrame, synth_df: pd.DataFrame, top_k: int = 10) -> None:
    if real_df.empty or synth_df.empty:
        return
    top_pairs = real_df["od_pair"].value_counts().head(top_k).index.tolist()
    print(f"[HoraOD][{tag}] top-{top_k} pares: {top_pairs}")
    peak_diffs = []
    for pair in top_pairs:
        r = real_df[real_df["od_pair"] == pair]
        s = synth_df[synth_df["od_pair"] == pair]
        real_h = _hour_histogram(r)
        synth_h = _hour_histogram(s)
        l1 = float(np.abs(synth_h - real_h).sum())
        tarde_gap = float(synth_h[15:19].sum() - real_h[15:19].sum())
        noite_gap = float(synth_h[20:24].sum() - real_h[20:24].sum())
        pr = _peak_hour(real_h)
        ps = _peak_hour(synth_h)
        peak_diffs.append(abs(((ps - pr + 12) % 24) - 12))
        print(
            f"  - {pair}: L1={l1:.3f} | Δtarde={tarde_gap:+.3f} | Δnoite={noite_gap:+.3f} | peak_real={pr} peak_synth={ps}"
        )
    if peak_diffs:
        print(
            f"[HoraOD][{tag}] peak_offset_mediana={np.median(peak_diffs):.3f} peak_offset_p90={np.percentile(peak_diffs,90):.3f}"
        )


# --------------------------------------------------------------------------- #
#                 AGREGAÇÃO DIÁRIA (10 DIAS CONSECUTIVOS)                     #
# --------------------------------------------------------------------------- #

def _aggregate_od_hour_day_daily(df: pd.DataFrame, *, days: int = 10, seed: int = 42) -> pd.DataFrame:
    """Seleciona até `days` dias consecutivos da janela de validação e agrega
    por data × od × hora_do_dia × dia_da_semana, salvando n_viagens.

    - Usa `seed` para escolher o ponto inicial de forma reprodutível.
    - Se houver menos de `days` dias disponíveis, usa todos os disponíveis.
    """
    if df.empty or "tpep_pickup_datetime" not in df.columns or "od_pair" not in df.columns:
        return pd.DataFrame(columns=["data", "od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio

    ts = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    dates = ts.dt.normalize()
    uniq = sorted(dates.dropna().unique())
    if not uniq:
        return pd.DataFrame(columns=["data", "od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio

    D = int(max(1, days))
    if len(uniq) <= D:
        chosen = uniq
    else:
        # janela deslizante com início determinado pela seed
        max_start = len(uniq) - D
        start = int(seed) % (max_start + 1)
        chosen = uniq[start : start + D]

    mask = dates.isin(chosen)
    sub = df.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=["data", "od", "hora_do_dia", "dia_da_semana", "n_viagens"])  # vazio

    # Deriva campos
    ts_sub = pd.to_datetime(sub["tpep_pickup_datetime"], errors="coerce")
    sub["data"] = ts_sub.dt.normalize().dt.date.astype(str)
    sub["hora_do_dia"] = ts_sub.dt.hour.astype(int)
    sub["dia_da_semana"] = ts_sub.dt.dayofweek.astype(int)
    sub["od"] = sub["od_pair"].astype(str)

    agg = (
        sub.groupby(["data", "od", "hora_do_dia", "dia_da_semana"]).size().reset_index(name="n_viagens")
    )
    agg = agg.sort_values(["data", "od", "dia_da_semana", "hora_do_dia"]).reset_index(drop=True)
    return agg


@torch.no_grad()
def _probe_decoder_conditioning(
    model: VAEMDN,
    od_artifacts: ODDiscretizerArtifacts,
    od_pairs: Iterable[str],
    *,
    device: torch.device | str = "cpu",
    repeats_z: int = 1,
) -> None:
    """Avalia se o decoder segue o condicionamento sincos_cond.

    Para cada par OD e horas 0..23, calcula a diferença circular média entre
    a hora condicionada e a hora média predita pelo MDN, tanto com z=0 quanto
    com z~N(0,1) (repeats_z vezes).
    """
    model = model.to(device).eval()
    lat_dim = model.latent_dim

    pairs = list(od_pairs)
    if not pairs:
        print("[CondProbe] Nenhum par OD fornecido.")
        return

    hours = np.arange(24, dtype=np.float32)
    angles = hours * (2 * np.pi / 24.0)
    sincos_24 = torch.stack(
        [torch.from_numpy(np.sin(angles)).to(device, dtype=torch.float32),
         torch.from_numpy(np.cos(angles)).to(device, dtype=torch.float32)], dim=1
    )  # [24,2]

    def _mdn_expected_hour(pi_logits, mdn_mu):
        # pi: [B,K]; mu: [B,K,2] -> média circular de horas em [0,24)
        pi = F.softmax(pi_logits, dim=-1)  # [B,K]
        mu = mdn_mu  # [B,K,2]
        # normaliza mu por componente para vetor unitário
        norm = torch.clamp(mu.norm(dim=-1, keepdim=True), min=1e-8)
        unit = mu / norm  # [B,K,2]
        # média vetorial ponderada por pi
        vx = (pi.unsqueeze(-1) * unit).sum(dim=1)  # [B,2]
        ang = torch.atan2(vx[:, 0], vx[:, 1])
        ang = torch.remainder(ang + 2 * np.pi, 2 * np.pi)
        hrs = ang * (24.0 / (2 * np.pi))  # [B]
        return hrs

    def _eval_for_z(z):
        diffs_all = []
        for pair in pairs:
            od_id = od_artifacts.od_to_id.get(pair)
            if od_id is None:
                continue
            pu = od_artifacts.pickup_id_for_od(od_id)
            do = od_artifacts.dropoff_id_for_od(od_id)
            pickup = torch.full((24,), pu, dtype=torch.long, device=device)
            dropoff = torch.full((24,), do, dtype=torch.long, device=device)
            pi_logits, mdn_mu, mdn_log_sigma = model.decode(z, pickup, dropoff, sincos_cond=sincos_24)
            pred_hours = _mdn_expected_hour(pi_logits, mdn_mu).detach().cpu().numpy()
            diffs = _circular_hour_diff(pred_hours, hours)
            diffs_all.append(diffs)
        if not diffs_all:
            return float("nan"), float("nan"), float("nan")
        diffs_all = np.concatenate(diffs_all)
        return float(np.mean(diffs_all)), float(np.percentile(diffs_all, 50)), float(np.percentile(diffs_all, 90))

    # z = 0
    z0 = torch.zeros(24, lat_dim, device=device)
    mean0, med0, p90_0 = _eval_for_z(z0)
    print(f"[CondProbe][z=0] meanΔh={mean0:.3f} medianΔh={med0:.3f} p90Δh={p90_0:.3f}")

    # z ~ N(0,1)
    means, meds, p90s = [], [], []
    for _ in range(max(1, repeats_z)):
        z = torch.randn(24, lat_dim, device=device)
        m, d, p = _eval_for_z(z)
        means.append(m); meds.append(d); p90s.append(p)
    print(
        f"[CondProbe][z~N] meanΔh={np.nanmean(means):.3f} "
        f"medianΔh={np.nanmean(meds):.3f} p90Δh={np.nanmean(p90s):.3f} repeats={repeats_z}"
    )


@torch.no_grad()
def _mdn_only_sample_hist(
    model: VAEMDN,
    *,
    n_samples: int,
    od_artifacts: ODDiscretizerArtifacts,
    od_distribution: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 131072,
) -> np.ndarray:
    """Gera amostras APENAS do MDN (sem prior de hora), com sincos_cond=None.

    Retorna histograma de 24 bins (probabilidades) dos horários decodificados.
    """
    model = model.to(device).eval()

    latent_dim = model.latent_dim
    od_distribution = od_distribution.astype(np.float64)
    od_distribution /= max(od_distribution.sum(), 1e-12)

    od_to_pickup = np.asarray(od_artifacts.od_to_pickup_id, dtype=np.int64)
    od_to_dropoff = np.asarray(od_artifacts.od_to_dropoff_id, dtype=np.int64)

    hour_hist = np.zeros(24, dtype=np.float64)

    for start in range(0, n_samples, batch_size):
        cur = min(batch_size, n_samples - start)
        sampled_od = np.random.choice(
            np.arange(len(od_distribution)), size=cur, p=od_distribution
        ).astype(np.int64)
        pickup = od_to_pickup[sampled_od]
        dropoff = od_to_dropoff[sampled_od]

        z = torch.randn(cur, latent_dim, device=device)
        pickup_t = torch.from_numpy(pickup).to(device)
        dropoff_t = torch.from_numpy(dropoff).to(device)

        # Decodifica sem condicionamento explícito de sin/cos
        pi_logits, mdn_mu, mdn_log_sigma = model.decode(
            z, pickup_t, dropoff_t, sincos_cond=None
        )
        samples = sample_from_mdn(pi_logits, mdn_mu, mdn_log_sigma).cpu().numpy()

        hours = decode_hour_from_sincos(samples[:, 0], samples[:, 1])
        bins = np.clip(np.floor(hours).astype(int), 0, 23)
        for b in bins:
            hour_hist[b] += 1.0

    return _normalize_histogram(hour_hist)


@torch.no_grad()
def _topk_od_prior_diagnostics(
    model: VAEMDN,
    df: pd.DataFrame,
    *,
    top_k: int = 10,
    device: torch.device | str = "cpu",
) -> None:
    """Imprime, para os top-k pares OD, métricas do prior p(h|OD) vs. real."""
    model = model.to(device).eval()
    top_pairs = df["od_pair"].value_counts().head(top_k).index.tolist()
    print(f"[PriorOD] top_k={top_k} pares={top_pairs}")

    for pair in top_pairs:
        mask = df["od_pair"] == pair
        sub = df.loc[mask]
        if sub.empty:
            continue
        pickup = torch.tensor(sub["pickup_id"].to_numpy(np.int64), device=device)
        dropoff = torch.tensor(sub["dropoff_id"].to_numpy(np.int64), device=device)
        logits = model.hour_logits(pickup, dropoff)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        pred = _normalize_histogram(probs.sum(axis=0))
        real = _hour_histogram(sub)

        l1 = float(np.abs(pred - real).sum())
        entropy = float(-(pred * np.log(pred + 1e-12)).sum())
        peak_p = int(np.argmax(pred))
        madrugada_p = float(pred[0:4].sum())
        tarde_p = float(pred[15:19].sum())
        noite_p = float(pred[20:24].sum())
        print(
            f"  - {pair}: L1={l1:.3f} entropy={entropy:.3f} peak={peak_p} "
            f"madrugada={madrugada_p:.3f} tarde={tarde_p:.3f} noite={noite_p:.3f}"
        )


def main() -> None:
    print("[Testes] Iniciando pipeline de diagnóstico...")

    # Treina e prepara dados (reutiliza o pipeline existente)
    # Flags:
    # - VAE_EXP: default=1 (treino experimental LIGADO por padrão)
    # - VAE_EXTRAS=1 habilita cenários NÃO-baseline de amostragem (bias/tau/blend, prior-direct, mdnmean)
    # - VAE_LATENT=1 habilita cenários com latent_bank (requer VAE_EXTRAS=1)
    run_exp = os.environ.get("VAE_EXP", "1") == "1"
    run_extras = os.environ.get("VAE_EXTRAS", "0") == "1"

    # Seed global (reduz variação entre execuções)
    seed = int(os.environ.get("VAE_SEED", "42"))
    set_global_seed(seed)
    overrides = None
    if run_exp:
        print("[Testes] Modo experimental ATIVO: scheduled conditioning + angle consistency + smoothing + marginal matching")
        # Permite override via variáveis de ambiente para facilitar varreduras sem editar o código
        def _getf(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except Exception:
                return float(default)
        def _geti(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except Exception:
                return int(default)

        overrides = dict(
            decoder_cond_mode=os.environ.get("DECODER_COND_MODE", "mix"),  # gt | prior_exp | mix
            decoder_mix_alpha_start=_getf("DECODER_MIX_ALPHA_START", 0.0),
            decoder_mix_alpha_end=_getf("DECODER_MIX_ALPHA_END", 0.9),
            hour_label_smoothing=_getf("HOUR_LABEL_SMOOTHING", 0.12),
            hour_smooth_radius=_geti("HOUR_SMOOTH_RADIUS", 1),
            prior_entropy_bonus=_getf("PRIOR_ENTROPY_BONUS", 0.001),
            prior_marginal_weight=_getf("PRIOR_MARGINAL_WEIGHT", 0.025),
            angle_consistency_weight=_getf("ANGLE_CONSISTENCY_WEIGHT", 0.9),
            sigma_penalty_weight=_getf("SIGMA_PENALTY_WEIGHT", 0.0),
        )
        print("[Testes][Overrides]", overrides)

    (
        model,
        train_df,
        val_df,
        hold_df,
        od_artifacts,
        od_distribution,
        _real_probs_train,
        _od_pairs_train,
        _latent_bank,
        used_params,
    ) = process_data(mdn_components=8, train_overrides=overrides)
    print(f"[Testes] Hiperparâmetros usados: {used_params}")

    # Salva (apenas) uma amostra do DataFrame FINAL processado usado no treino do VAE.
    # Importante: este snapshot é feito após toda a discretização/engenharia do pipeline.
    out_csv = SCRIPT_DIR / "train_processed_sample_20rows.csv"
    train_input_cols = ["sin_hr", "cos_hr", "pickup_id", "dropoff_id"]
    train_df[train_input_cols].head(20).to_csv(out_csv, index=False)
    print(f"[Testes] Snapshot do treino processado salvo em: {out_csv} | cols={train_input_cols}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Histograma real (train/val)
    real_train = _hour_histogram(train_df)
    real_val = _hour_histogram(val_df)
    _print_hist_24("real:train", real_train)
    _print_hist_24("real:val", real_val)

    # Calibração do prior (train/val)
    prior_train = _aggregate_prior_hist(model, train_df, device=device)
    prior_val = _aggregate_prior_hist(model, val_df, device=device)
    _compare_hists("PriorCalib:train", real_train, prior_train)
    _compare_hists("PriorCalib:val", real_val, prior_val)

    # Varredura de temperatura no prior (val)
    for tau in [1.0, 1.3, 1.6, 2.0]:
        prior_tau = _aggregate_prior_hist(model, val_df, device=device, temperature=tau)
        _compare_hists(f"PriorTempSweep:val:tau={tau}", real_val, prior_tau)

    # Diagnóstico por par OD do prior (top-10 em val)
    _topk_od_prior_diagnostics(model, val_df, top_k=10, device=device)

    # ---------------------------------------------------------------
    # Novo: exporta validação diária (até 10 dias consecutivos)
    # ---------------------------------------------------------------
    try:
        seed_env = int(os.environ.get("VAE_SEED", "42"))
        daily_agg = _aggregate_od_hour_day_daily(val_df, days=10, seed=seed_env)
        out_dir = SCRIPT_DIR / "save_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "od_hora_dia_validacao_10dias.csv"
        daily_agg.to_csv(out_path, index=False)
        total_val = int(daily_agg["n_viagens"].sum()) if not daily_agg.empty else 0
        print(f"[CSV] Validação diária (10 dias) salva em: {out_path} | total_n_viagens={total_val}")
    except Exception as e:
        print(f"[CSV] Falha ao gerar validação diária: {e}")

    # MDN-only: gera distribuição de horas sem prior (diagnóstico, não afeta baseline)
    mdn_only_hist = _mdn_only_sample_hist(
        model,
        n_samples=min(300_000, max(100_000, int(len(train_df) * 0.25))),
        od_artifacts=od_artifacts,
        od_distribution=od_distribution,
        device="cpu",  # leve para não ocupar GPU
    )
    _compare_hists("MDNonly:val_ref", real_val, mdn_only_hist)

    # ---------------------------------------------------------------
    # Testes de amostragem com ajustes (EXTRAS): bias, temperatura e blend
    # ---------------------------------------------------------------
    if run_extras:
        print("[Testes] Preparando bias global por hora a partir do train...")
        eps = 1e-8
        prior_train = _aggregate_prior_hist(model, train_df, device=device)
        real_train = _hour_histogram(train_df)
        hour_bias = np.log(real_train + eps) - np.log(prior_train + eps)
        _print_hist_24("Bias:prior_train", prior_train)
        _print_hist_24("Bias:real_train", real_train)
        print(f"[Testes] hour_bias (log-ratio) resumo: min={hour_bias.min():.4f} max={hour_bias.max():.4f} std={hour_bias.std():.4f}")

    # Tamanho da amostra para testes de geração
    n_synth_test = min(400_000, max(150_000, int(len(train_df) * 0.25)))

    def _sample_and_compare(
        tag: str,
        *,
        use_latent_bank: bool = False,
        plot_out_path: Path | None = None,
        real_hist_plot: np.ndarray | None = None,
        prior_val_hist: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        latent_bank = None
        if use_latent_bank:
            print("[Testes] Construindo latent_bank do train para amostragem...")
            latent_bank = build_latent_bank(
                model,
                train_df[["sin_hr", "cos_hr", "pickup_id", "dropoff_id", "od_id"]].copy(),
                device=device,
                batch_size=65536,
            )
        df_syn = sample_vae_mdn(
            model,
            n_samples=n_synth_test,
            od_artifacts=od_artifacts,
            od_distribution=od_distribution,
            device="cpu",
            latent_bank=latent_bank,
            **kwargs,
        )
        # Calcula histograma de horas
        syn_hist = _hour_histogram(df_syn)
        _compare_hists(tag, real_val, syn_hist)
        # Acurácia de alinhamento entre hora_target e hora_do_dia
        acc0, acc1, md = _hour_alignment_metrics(df_syn)
        print(f"[Align][{tag}] acc0={acc0:.3f} acc±1={acc1:.3f} mean_circ_diff={md:.3f}")

        # Plot opcional (sobrescreve o arquivo em cada execução)
        if plot_out_path is not None and real_hist_plot is not None:
            try:
                _plot_hour_histogram_compare(
                    real_hist=real_hist_plot,
                    synth_hist=syn_hist,
                    out_path=plot_out_path,
                    title=f"Hora do dia — Real vs Sintética ({tag})",
                )
                print(f"[Plot] Histograma salvo em: {plot_out_path}")
            except Exception as e:
                print(f"[Plot] Falha ao gerar gráfico '{plot_out_path}': {e}")

        # Métrica OD simples no top-10 da validação
        top_pairs = val_df["od_pair"].value_counts().head(10).index
        real_od = (val_df["od_pair"].value_counts().reindex(top_pairs, fill_value=0))
        real_od = (real_od / max(real_od.sum(), 1))
        syn_od = (df_syn["od_pair"].value_counts().reindex(top_pairs, fill_value=0))
        syn_od = (syn_od / max(syn_od.sum(), 1))
        od_linf = float((syn_od - real_od).abs().max())
        print(f"[Compare][{tag}:OD] max|Δ|_top10={od_linf:.4f}")
        # Hora por top OD (val)
        _top_od_hour_metrics(tag, val_df, df_syn, top_k=int(os.environ.get("VAE_TOPK","10")))

        # Investigação por subconjunto de horas onde real > sintético
        investigate_this = (tag == "Sampling:baseline") or ("mdnmean" in tag)
        if investigate_this:
            subset_hours = [5, 6, 7, 21, 22]
            real_hist = _hour_histogram(val_df)
            target_hist = _hour_target_histogram(df_syn)
            synth_hist = syn_hist
            delta = real_hist - synth_hist
            subset_mass_real = float(real_hist[subset_hours].sum())
            subset_mass_target = float(target_hist[subset_hours].sum())
            subset_mass_synth = float(synth_hist[subset_hours].sum())
            subset_mass_prior_val = None
            if prior_val_hist is not None:
                subset_mass_prior_val = float(prior_val_hist[subset_hours].sum())
            print(
                f"[Investigate][SubsetMass] hours={subset_hours} | real={subset_mass_real:.4f} "
                f"target(prior_train_mix)={subset_mass_target:.4f} "
                + (f"prior_val={subset_mass_prior_val:.4f} " if subset_mass_prior_val is not None else "")
                + f"synth={subset_mass_synth:.4f}"
            )
            for h in subset_hours:
                pv = prior_val_hist[h] if prior_val_hist is not None else float('nan')
                print(
                    f"  - h={h:02d}: real={real_hist[h]:.4f} target={target_hist[h]:.4f} "
                    + (f"prior_val={pv:.4f} " if prior_val_hist is not None else "")
                    + f"synth={synth_hist[h]:.4f} Δ(real-synth)={delta[h]:+.4f}"
                )
            _circular_err_stats_for_targets(df_syn, subset_hours)
            _confusion_for_targets(df_syn, subset_hours, topk=5)

            # Top pares OD que mais perdem massa no subconjunto de horas
            _subset_od_differences(
                tag,
                model,
                od_artifacts,
                val_df,
                df_syn,
                subset_hours=subset_hours,
                top_k=int(os.environ.get("VAE_INVEST_TOPK", "15")),
                device="cpu",
            )

    print("[Testes] Amostragem baseline (sem ajustes)...")
    # Salva o comparativo de histogramas (sobrescreve a cada run)
    baseline_plot_path = SCRIPT_DIR / "hora_global_comparativo.png"
    _sample_and_compare(
        "Sampling:baseline",
        plot_out_path=baseline_plot_path,
        real_hist_plot=real_val,
        prior_val_hist=prior_val,
    )
    # Blocos com latent_bank são opcionais (habilite com VAE_EXTRAS=1 e VAE_LATENT=1)
    use_lat = run_extras and (os.environ.get("VAE_LATENT", "0") == "1")
    if use_lat:
        print("[Testes] Amostragem baseline com latent_bank...")
        _sample_and_compare("Sampling:baseline+latent", use_latent_bank=True)

    if run_extras:
        print("[Testes] Amostragem com bias global...")
        _sample_and_compare("Sampling:bias", hour_bias=hour_bias)
        if use_lat:
            print("[Testes] Amostragem com bias global + latent_bank...")
            _sample_and_compare("Sampling:bias+latent", hour_bias=hour_bias, use_latent_bank=True)

        print("[Testes] Amostragem com bias + temperatura τ=1.6...")
        _sample_and_compare("Sampling:bias+tau1.6", hour_bias=hour_bias, hour_temperature=1.6)
        if use_lat:
            print("[Testes] Amostragem com bias + τ=1.6 + latent_bank...")
            _sample_and_compare("Sampling:bias+tau1.6+latent", hour_bias=hour_bias, hour_temperature=1.6, use_latent_bank=True)

        print("[Testes] Amostragem com bias + τ=1.6 + blend α=0.1 com real(val)...")
        _sample_and_compare(
            "Sampling:bias+tau1.6+blend0.1",
            hour_bias=hour_bias,
            hour_temperature=1.6,
            hour_blend_alpha=0.10,
            global_hour_hist=real_val,
        )

    # Variantes usando média do MDN (sem ruído angular)
    if run_extras:
        print("[Testes] Amostragem baseline usando média do MDN...")
        _sample_and_compare(
            "Sampling:baseline_mdnmean",
            use_mdn_mean=True,
            prior_val_hist=prior_val,
        )

        print("[Testes] Amostragem com bias + τ=1.6 usando média do MDN...")
        _sample_and_compare(
            "Sampling:bias+tau1.6_mdnmean",
            hour_bias=hour_bias,
            hour_temperature=1.6,
            use_mdn_mean=True,
        )
        if use_lat:
            print("[Testes] Amostragem com bias + τ=1.6 + blend α=0.1 + latent_bank...")
            _sample_and_compare(
                "Sampling:bias+tau1.6+blend0.1+latent",
                hour_bias=hour_bias,
                hour_temperature=1.6,
                hour_blend_alpha=0.10,
                global_hour_hist=real_val,
                use_latent_bank=True,
            )

    # ---------------------------------------------------------------
    # Teste alternativo: usar horas do prior diretamente para sin/cos
    # ---------------------------------------------------------------
    if run_extras:
        print("[Testes] Amostragem prior-direct (usa horas do prior p/ sin/cos)...")
        _sample_and_compare("Sampling:prior-direct", use_prior_hours_directly=True)

        print("[Testes] prior-direct com bias + τ=1.6...")
        _sample_and_compare(
            "Sampling:prior-direct+bias+tau1.6",
            hour_bias=hour_bias,
            hour_temperature=1.6,
            use_prior_hours_directly=True,
        )

        print("[Testes] prior-direct com bias + τ=1.6 + blend α=0.1 (val)...")
        _sample_and_compare(
            "Sampling:prior-direct+bias+tau1.6+blend0.1",
            hour_bias=hour_bias,
            hour_temperature=1.6,
            hour_blend_alpha=0.10,
            global_hour_hist=real_val,
            use_prior_hours_directly=True,
        )

    # ---------------------------------------------------------------
    # Probing: o decoder está sensível a sincos_cond?
    # ---------------------------------------------------------------
    if run_extras:
        print("[Testes] Probing do condicionamento do decoder (sincos_cond→hora)...")
        top_pairs_probe = val_df["od_pair"].value_counts().head(6).index.tolist()
        _probe_decoder_conditioning(model, od_artifacts, top_pairs_probe, device=device, repeats_z=3)

    # Sumariza achados imediatos
    print("[Resumo]")
    print(
        "- Compare PriorCalib vs. real: se L1/JSD altos e madrugada/tarde deslocados, o problema está no prior p(h|OD)."
    )
    print(
        "- Veja PriorTempSweep: se τ>1 reduz L1 substancialmente, o prior está overconfident (temperatura ajuda)."
    )
    print(
        "- Veja MDNonly: se fica próximo do real, MDN não é o gargalo; se também desvia, revisar MDN/condicionamento."
    )


if __name__ == "__main__":
    main()
