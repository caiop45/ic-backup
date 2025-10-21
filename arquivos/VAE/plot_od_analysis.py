"""Geração de gráficos comparativos para distribuições OD e horários."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.helpers import decode_hour_from_sincos

PALETTE = {
    "real": "#1b9e77",  # verde
    "synth": "#377eb8",  # azul
    "diff": "#e41a1c",  # vermelho
}


def _ensure_hours(df: pd.DataFrame) -> pd.Series:
    if "hora_do_dia" in df.columns:
        return df["hora_do_dia"].astype(int)
    hours = decode_hour_from_sincos(df["sin_hr"], df["cos_hr"])
    return hours.astype(int)


def _top_pairs(real_df: pd.DataFrame, synth_df: pd.DataFrame, k: int = 10) -> Iterable[str]:
    real_top20 = real_df["od_pair"].value_counts().head(20)
    synth_top20 = synth_df["od_pair"].value_counts().head(20)
    common = real_top20.index.intersection(synth_top20.index)
    top = real_top20.loc[common]
    return top.head(k).index.tolist()


def _plot_hour_distribution(real_df: pd.DataFrame, synth_df: pd.DataFrame, out_path: Path) -> None:
    real_hours = _ensure_hours(real_df)
    synth_hours = _ensure_hours(synth_df)

    real_counts = real_hours.value_counts().sort_index()
    synth_counts = synth_hours.value_counts().sort_index()

    idx = pd.Index(range(24))
    real_probs = (real_counts.reindex(idx, fill_value=0) / real_counts.sum()) * 100
    synth_probs = (synth_counts.reindex(idx, fill_value=0) / synth_counts.sum()) * 100
    diff = synth_probs - real_probs

    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(idx))
    plt.bar(x - width / 2, real_probs, width=width, color=PALETTE["real"], label="Real (%)")
    plt.bar(x + width / 2, synth_probs, width=width, color=PALETTE["synth"], label="Sintético (%)")
    plt.plot(x, diff, color=PALETTE["diff"], label="Diferença (pp)")
    plt.xticks(x, idx)
    plt.xlabel("Hora do dia")
    plt.ylabel("Participação (%)")
    plt.title("Distribuição global de hora do dia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_hour_by_pair(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    pairs: Iterable[str],
    out_path: Path,
    *,
    synth_scale: float = 1.0,
    synth_label: str = "Sintético",
) -> None:
    real_hours = _ensure_hours(real_df)
    synth_hours = _ensure_hours(synth_df)

    n_pairs = len(pairs)
    cols = 5
    rows = int(np.ceil(n_pairs / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax_idx, pair in enumerate(pairs):
        ax = axes[ax_idx]
        mask_real = real_df["od_pair"] == pair
        mask_synth = synth_df["od_pair"] == pair
        real_counts = real_hours[mask_real].value_counts().sort_index()
        synth_counts = synth_hours[mask_synth].value_counts().sort_index()
        idx = pd.Index(range(24))
        real_vals = real_counts.reindex(idx, fill_value=0)
        synth_vals = synth_counts.reindex(idx, fill_value=0).astype(float) * synth_scale
        width = 0.35
        x = np.arange(len(idx))
        ax.bar(x - width / 2, real_vals, width=width, color=PALETTE["real"], label="Real")
        ax.bar(x + width / 2, synth_vals, width=width, color=PALETTE["synth"], label=synth_label)
        ax.set_title(pair, fontsize=9)
        ax.set_xticks([0, 6, 12, 18, 23])

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.suptitle("Distribuição horária – Top pares OD", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path)
    plt.close(fig)


def _plot_top10_counts(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    pairs: Iterable[str],
    out_path: Path,
    *,
    synth_scale: float = 1.0,
    synth_label: str = "Sintético",
) -> None:
    real_counts = real_df["od_pair"].value_counts()
    synth_counts = synth_df["od_pair"].value_counts()
    data = pd.DataFrame(
        {
            "real": real_counts.reindex(pairs, fill_value=0),
            "synth": synth_counts.reindex(pairs, fill_value=0).astype(float) * synth_scale,
        }
    )
    diff = data["synth"] - data["real"]

    x = np.arange(len(pairs))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, data["real"], width=width, color=PALETTE["real"], label="Real")
    plt.bar(x + width / 2, data["synth"], width=width, color=PALETTE["synth"], label=synth_label)
    plt.bar(x, diff, width=0.15, color=PALETTE["diff"], label="Diferença")
    plt.xticks(x, pairs, rotation=45, ha="right")
    plt.ylabel("Contagem")
    plt.title("Top 10 pares OD – contagens")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _prepare_pivot(df: pd.DataFrame, value_col: str = "count", top_n: int = 20) -> pd.DataFrame:
    counts = df.groupby(["pickup_location", "dropoff_location"]).size().rename("count")
    top_pairs = counts.sort_values(ascending=False).head(top_n)
    top_pickups = top_pairs.groupby(level=0).sum().sort_values(ascending=False).head(top_n).index
    top_dropoffs = top_pairs.groupby(level=1).sum().sort_values(ascending=False).head(top_n).index
    filtered = counts.loc[(counts.index.get_level_values(0).isin(top_pickups)) & (counts.index.get_level_values(1).isin(top_dropoffs))]
    pivot = (
        filtered
        .rename("count")
        .unstack(fill_value=0)
        .reindex(index=top_pickups, columns=top_dropoffs, fill_value=0)
    )
    return pivot / max(pivot.sum().sum(), 1)


def _plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, cmap="Blues", linewidths=0.5)
    plt.title(title)
    plt.xlabel("Destino")
    plt.ylabel("Origem")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_cdf(real_df: pd.DataFrame, synth_df: pd.DataFrame, out_path: Path) -> None:
    real_counts = real_df["od_pair"].value_counts().sort_values(ascending=False).reset_index(drop=True)
    synth_counts = synth_df["od_pair"].value_counts().sort_values(ascending=False).reset_index(drop=True)
    n = max(len(real_counts), len(synth_counts))
    real_cdf = (real_counts.reindex(range(n), fill_value=0).cumsum() / real_counts.sum())
    synth_cdf = (synth_counts.reindex(range(n), fill_value=0).cumsum() / synth_counts.sum())
    diff = synth_cdf - real_cdf

    x = np.arange(1, n + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x, real_cdf, color=PALETTE["real"], label="Real")
    plt.plot(x, synth_cdf, color=PALETTE["synth"], label="Sintético")
    plt.plot(x, diff, color=PALETTE["diff"], label="Diferença")
    plt.xlabel("Rank do par OD")
    plt.ylabel("Participação acumulada")
    plt.title("CDF por rank de pares OD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_abs_diff(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    out_path: Path,
    *,
    synth_scale: float = 1.0,
) -> None:
    real_counts = real_df["od_pair"].value_counts()
    synth_counts = synth_df["od_pair"].value_counts()
    union_pairs = real_counts.add(synth_counts, fill_value=0).sort_values(ascending=False).head(20).index
    scaled_synth = synth_counts.reindex(union_pairs, fill_value=0).astype(float) * synth_scale
    abs_diff = (scaled_synth - real_counts.reindex(union_pairs, fill_value=0)).abs()

    plt.figure(figsize=(12, 6))
    plt.bar(union_pairs, abs_diff, color=PALETTE["diff"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("|Δ contagem|")
    plt.title("Diferenças absolutas por par OD (top 20)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def generate_all_plots(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    output_dir: Path,
    *,
    tag: str | None = None,
    synth_scale: float = 1.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs_top10 = _top_pairs(real_df, synth_df, k=10)

    suffix = f"_{tag}" if tag else ""
    synth_label = "Sintético" if np.isclose(synth_scale, 1.0) else "Sintético (normalizado)"

    _plot_hour_distribution(
        real_df,
        synth_df,
        output_dir / f"hora_global{suffix}.png",
    )
    _plot_hour_by_pair(
        real_df,
        synth_df,
        pairs_top10,
        output_dir / f"hora_por_top10_od{suffix}.png",
        synth_scale=synth_scale,
        synth_label=synth_label,
    )
    _plot_top10_counts(
        real_df,
        synth_df,
        pairs_top10,
        output_dir / f"od_top10_barras{suffix}.png",
        synth_scale=synth_scale,
        synth_label=synth_label,
    )

    real_pivot = _prepare_pivot(real_df)
    synth_pivot = _prepare_pivot(synth_df)
    _plot_heatmap(real_pivot, "Matriz OD normalizada – Real", output_dir / f"od_heatmap_real{suffix}.png")
    _plot_heatmap(synth_pivot, "Matriz OD normalizada – Sintético", output_dir / f"od_heatmap_synth{suffix}.png")

    _plot_cdf(real_df, synth_df, output_dir / f"od_cdf_rank{suffix}.png")
    _plot_abs_diff(real_df, synth_df, output_dir / f"od_diferencas_top20{suffix}.png", synth_scale=synth_scale)
