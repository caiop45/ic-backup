import math
import seaborn as sns
import matplotlib.pyplot as plt
from config import SAVE_DIR
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np 

def generate_plots(df):
    """
    Boxplots por métrica + curvas MAE vs R².
    (Salva arquivos em SAVE_DIR)
    """
    if df.empty:
        print("DataFrame de resultados vazio – sem gráficos.")
        return

    sns.set_style("whitegrid")
    df.to_csv(f"{SAVE_DIR}resultados_metricas.csv", index=False)

    order = ["real", "synthetic", "real+synthetic"]
    for metrica in df["metrica"].unique():
        for nc in sorted(df["nc"].unique()):
            sub_nc = df[(df["metrica"] == metrica) & (df["nc"] == nc)]
            if sub_nc.empty:
                continue

            ep_list = sorted(sub_nc["epochs"].unique())
            rows = math.ceil(len(ep_list) / 3)
            fig, axs = plt.subplots(rows, 3, figsize=(19.5, 5.5 * rows), squeeze=False)
            axs = axs.flatten()

            for i, ep in enumerate(ep_list):
                ax      = axs[i]
                sub_ep  = sub_nc[sub_nc["epochs"] == ep]
                sns.boxplot(x="tipo_dado", y="valor", order=order, data=sub_ep, ax=ax)
                ax.set_title(f"nc={nc} | epochs={ep}")
                ax.set_xlabel("")
                ax.set_ylabel(metrica)
            for j in range(i + 1, len(axs)):
                axs[j].axis("off")

            fig.suptitle(f"{metrica} – nc={nc}", fontsize=14, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(f"{SAVE_DIR}nc_{nc}_{metrica}.png", bbox_inches="tight")
            plt.close(fig)

    # MAE vs R²
    try:
        piv = df.pivot_table(index=["tipo_dado", "nc", "epochs", "seed"],
                             columns="metrica", values="valor").reset_index()
        if {"R²", "MAE"}.issubset(piv.columns):
            avg = piv.groupby(["tipo_dado", "epochs", "nc"], as_index=False)[["R²", "MAE"]].mean()
            for t in avg["tipo_dado"].unique():
                sub = avg[avg["tipo_dado"] == t]
                plt.figure(figsize=(10, 6))
                for ep in sorted(sub["epochs"].unique()):
                    sub_ep = sub[sub["epochs"] == ep].sort_values("nc")
                    plt.plot(sub_ep["R²"], sub_ep["MAE"], marker="o", label=f"epochs={ep}")
                plt.xlabel("R²"); plt.ylabel("MAE"); plt.title(f"MAE vs R² – {t}")
                plt.legend(title="Épocas"); plt.grid(True); plt.tight_layout()
                plt.savefig(f"{SAVE_DIR}{t.replace('+', 'plus')}_MAE_vs_R2.png", bbox_inches="tight")
                plt.close()
    except Exception as e:
        print(f"Falha ao gerar MAE vs R²: {e}")


def plot_hourly_trip_comparison(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    run_idx: int,
    out_dir: Union[str, Path],
    *,
    hour_col: str = "hora_do_dia",
    trips_col: str = "num_viagens",
    dpi: int = 150,
) -> None:
    """
    Gera e salva um gráfico de barras lado a lado comparando o número total
    de viagens por hora entre dados reais e sintéticos.

    Parameters
    ----------
    real_df : DataFrame
        DataFrame com colunas `hour_col` e `trips_col` para os dados reais.
    synth_df : DataFrame
        DataFrame com colunas `hour_col` e `trips_col` para os dados sintéticos.
    run_idx : int
        Índice (1-based) da execução atual — usado no título e no nome do arquivo.
    out_dir : str ou Path
        Diretório onde o PNG será salvo.
    hour_col : str, opcional
        Nome da coluna que contém a hora do dia (0-23). Default = "hora_do_dia".
    trips_col : str, opcional
        Nome da coluna que contém o total de viagens (uma linha = 1 viagem). Default = "num_viagens".
    dpi : int, opcional
        Resolução do arquivo salvo.
    """
    # soma viagens por hora
    real_counts = real_df.groupby(hour_col)[trips_col].sum()
    synth_counts = synth_df.groupby(hour_col)[trips_col].sum()

    # garante todas as 24 h no eixo X
    horas = np.arange(24)
    real_vals = real_counts.reindex(horas, fill_value=0).values
    synth_vals = synth_counts.reindex(horas, fill_value=0).values

    # gráfico
    largura = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(horas - largura / 2, synth_vals, width=largura, label="Sintético")
    ax.bar(horas + largura / 2, real_vals,  width=largura, label="Real")

    # anotação da diferença percentual
    max_altura = max(real_vals.max(), synth_vals.max())
    for h, s_val, r_val in zip(horas, synth_vals, real_vals):
        texto = "∞%" if r_val == 0 else f"{(s_val - r_val) / r_val * 100:+.1f}%"
        y_pos = max(s_val, r_val) + 0.05 * max_altura
        ax.text(h, y_pos, texto, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(horas)
    ax.set_xlabel("Hora do dia")
    ax.set_ylabel("Número total de viagens")
    ax.set_title(f"Comparação de viagens por hora — Execução {run_idx}")
    ax.legend()
    plt.tight_layout()

    # salva
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"viagens_por_hora_run_{run_idx}.png"
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

def plot_random_pair_heatmaps(
    real_data: pd.DataFrame,
    synth_data: pd.DataFrame,
    run_number: int,
    save_dir: str,
    num_pairs: int = 20,
    random_seed: Optional[int] = None,
) -> None:
    """
    Gera e salva um par de heatmaps comparando a contagem de viagens entre dados
    reais e sintéticos para um número aleatório de pares de localizações (PU/DO).

    Args:
        real_data (pd.DataFrame): DataFrame com os dados reais.
            Deve conter as colunas 'PULocationID', 'DOLocationID' e 'num_viagens'.
        synth_data (pd.DataFrame): DataFrame com os dados sintéticos.
            Deve conter as colunas 'PULocationID', 'DOLocationID' e 'num_viagens'.
        run_number (int): O número da execução atual, usado para títulos e nome do arquivo.
        save_dir (str): O diretório onde a imagem do gráfico será salva.
        num_pairs (int, optional): O número de pares (PU, DO) aleatórios a serem amostrados.
            Default é 20.
        random_seed (Optional[int], optional): Semente para o gerador de números aleatórios
            para garantir a reprodutibilidade da amostragem. Default é None.
    """
    # 1) Pivot completo dos dados reais e sintéticos
    pivot_real = pd.pivot_table(
        real_data,
        index="PULocationID",
        columns="DOLocationID",
        values="num_viagens",
        aggfunc="sum",
        fill_value=0,
    )

    pivot_synth = pd.pivot_table(
        synth_data,
        index="PULocationID",
        columns="DOLocationID",
        values="num_viagens",
        aggfunc="sum",
        fill_value=0,
    )

    # 2) Selecionar `num_pairs` pares (PU, DO) aleatórios a partir dos dados sintéticos
    todas_tuplas = synth_data[["PULocationID", "DOLocationID"]].drop_duplicates()
    if len(todas_tuplas) < num_pairs:
        print(f"Aviso: O número de pares únicos ({len(todas_tuplas)}) é menor que `num_pairs` ({num_pairs}). Usando todos os pares.")
        num_pairs = len(todas_tuplas)
        
    sample_pairs = todas_tuplas.sample(n=num_pairs, random_state=random_seed)

    # 3) Extrair os conjuntos de PUs e DOs
    pu_ids_selecionados = sample_pairs["PULocationID"].unique().tolist()
    do_ids_selecionados = sample_pairs["DOLocationID"].unique().tolist()

    # 4) Fatiar o pivot para manter só os PUs e DOs de interesse
    pivot_real_sub = pivot_real.reindex(
        index=pu_ids_selecionados, columns=do_ids_selecionados, fill_value=0
    )
    pivot_synth_sub = pivot_synth.reindex(
        index=pu_ids_selecionados, columns=do_ids_selecionados, fill_value=0
    )

    # 5) Plotar os dois heatmaps
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), constrained_layout=True)
    
    # Define a escala de cor com base nos dados reais
    vmin = pivot_real_sub.values.min()
    vmax = pivot_real_sub.values.max()

    # Heatmap (dados reais)
    im0 = axes[0].imshow(
        pivot_real_sub.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title(f"Dados Reais (run {run_number}) — {num_pairs} pares aleatórios")
    axes[0].set_xlabel("DOLocationID")
    axes[0].set_ylabel("PULocationID")
    axes[0].set_xticks(range(len(pivot_real_sub.columns)))
    axes[0].set_xticklabels(pivot_real_sub.columns, rotation=90, fontsize=6)
    axes[0].set_yticks(range(len(pivot_real_sub.index)))
    axes[0].set_yticklabels(pivot_real_sub.index, fontsize=6)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("num_viagens (real)")

    # Heatmap (dados sintéticos)
    im1 = axes[1].imshow(
        pivot_synth_sub.values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title(f"Dados Sintéticos (run {run_number}) — {num_pairs} pares aleatórios")
    axes[1].set_xlabel("DOLocationID")
    axes[1].set_yticks([])
    axes[1].set_xticks(range(len(pivot_synth_sub.columns)))
    axes[1].set_xticklabels(pivot_synth_sub.columns, rotation=90, fontsize=6)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("num_viagens (sintético)")

    # 6) Salvar a figura
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f"heatmap_{num_pairs}pares_run_{run_number}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def boxplot_model_eval(
    metrics_dict: dict[str, dict[str, float]],
    metric_names: list[str],
    suptitle: str,
    save_path: str | Path,
) -> None:
    """
    metrics_dict  = {
        "Real":            {"R²": .., "SMAPE": .., "MAE": ..},
        "Sintético":       {...},
        "Real + Sintético":{...}
    }
    metric_names = ["R²", "SMAPE", "MAE"]
    """
    n_cols = 2
    n_rows = math.ceil(len(metric_names) / n_cols)

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(10, 4 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    labels = list(metrics_dict.keys())  # ordem das barras

    for idx, metric in enumerate(metric_names):
        ax = axes_flat[idx]
        vals = [metrics_dict[label][metric] for label in labels]

        bars = ax.bar(labels, vals)

        # anotações
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_title(metric)
        ax.set_ylim(bottom=0)

    # remove subplots vazios (se nº de métricas for ímpar)
    for ax in axes_flat[len(metric_names) :]:
        fig.delaxes(ax)

    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)