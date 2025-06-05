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