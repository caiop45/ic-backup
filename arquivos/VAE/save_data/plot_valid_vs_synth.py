"""
Gera gráficos comparativos simples entre validação e sintético usando
os CSVs agregados por OD × hora × dia.

Entradas (nesta pasta):
- od_hora_dia_validacao.csv
- od_hora_dia_sintetico.csv

Saídas:
- ../graficos/od_top15_participacao_validacao_vs_sintetico.png
- ../graficos/od_top15_por_hora/od_<slug>_distrib_hora_validacao_vs_sintetico.png (15 arquivos)
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend não interativo
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
VAE_DIR = SCRIPT_DIR.parent
GRAFICOS_DIR = VAE_DIR / "graficos"

# Número de ODs do topo a considerar nos gráficos/CSVs
TOP_N = 100


def _slugify(text: str, maxlen: int = 90) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if maxlen > 0 else s


def _load_inputs():
    val_path = SCRIPT_DIR / "od_hora_dia_validacao.csv"
    syn_path = SCRIPT_DIR / "od_hora_dia_sintetico.csv"
    if not val_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {val_path}")
    if not syn_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {syn_path}")
    val = pd.read_csv(val_path)
    syn = pd.read_csv(syn_path)
    # Garantias mínimas de schema
    req_cols = {"od", "hora_do_dia", "dia_da_semana", "n_viagens"}
    if not req_cols.issubset(val.columns):
        raise ValueError(f"CSV de validação não possui colunas esperadas: {req_cols}")
    if not req_cols.issubset(syn.columns):
        raise ValueError(f"CSV sintético não possui colunas esperadas: {req_cols}")
    return val, syn


def _ensure_dirs(top_n: int = TOP_N):
    GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)
    (GRAFICOS_DIR / f"od_top{int(top_n)}_por_hora").mkdir(parents=True, exist_ok=True)


def _annotate_bars(ax: plt.Axes, rects, fmt: str = "{:.1%}", fontsize: int = 8, dy: float = 0.01) -> None:
    """Escreve o valor acima de cada barra.

    dy é um pequeno deslocamento vertical para evitar sobreposição com o topo da barra.
    """
    for r in rects:
        h = r.get_height()
        if pd.isna(h):
            continue
        x = r.get_x() + r.get_width() / 2
        y = (h + dy) if h > 0 else dy
        ax.text(x, y, fmt.format(h), ha="center", va="bottom", fontsize=fontsize)


def plot_participacao_topN(val: pd.DataFrame, syn: pd.DataFrame, top_n: int = TOP_N) -> list[str]:
    # Totais por OD (para ordenar) e totais globais (denominador em todo o CSV)
    val_tot = val.groupby("od")["n_viagens"].sum().sort_values(ascending=False)
    top_ods = val_tot.head(int(top_n)).index.tolist()

    if len(top_ods) == 0:
        print(f"[INFO] Nenhuma OD encontrada em validação para top{top_n}.")
        return []

    syn_tot = syn.groupby("od")["n_viagens"].sum()

    # Denominadores: total de viagens do CSV completo (sem filtrar pelas top-N)
    val_total_all = float(val["n_viagens"].sum())
    syn_total_all = float(syn["n_viagens"].sum())
    print(f"[INFO] Totais | validação: {int(val_total_all)} | sintético: {int(syn_total_all)}")

    # Participações relativas ao total global de cada conjunto
    val_den = (val_tot.reindex(top_ods) / max(val_total_all, 1)).fillna(0.0)
    syn_den = (syn_tot.reindex(top_ods) / max(syn_total_all, 1)).fillna(0.0)

    # Soma das Top-N em % do total
    val_topn = float(val_den.sum())
    syn_topn = float(syn_den.sum())
    print(f"[INFO] Top{top_n} participação | validação: {val_topn:.2%} | sintético: {syn_topn:.2%}")

    x = np.arange(len(top_ods))
    width = 0.42

    # Dobrar o espaço horizontal em relação ao anterior (0.8 → 1.6; 10 → 20)
    fig_width = max(20, len(top_ods) * 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    r1 = ax.bar(x - width / 2, val_den.values, width=width, label="Validação")
    r2 = ax.bar(x + width / 2, syn_den.values, width=width, label="Sintético")
    ax.set_xticks(x)
    ax.set_xticklabels(top_ods, rotation=60, ha="right")
    ax.set_ylabel("Participação (%)")
    ax.set_title(f"Top {top_n} OD – Participação (Validação × Sintético)")
    ax.legend()
    # Espaço para rótulos acima das barras
    ymax = float(max(val_den.max(), syn_den.max())) if len(val_den) else 0.0
    ax.set_ylim(0, min(1.05, ymax + 0.06))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    _annotate_bars(ax, r1)
    _annotate_bars(ax, r2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)  # garante espaço para rótulos do eixo X

    out = GRAFICOS_DIR / f"od_top{int(top_n)}_participacao_validacao_vs_sintetico.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] Gráfico salvo: {out}")

    # CSV com os valores do gráfico Top-N (nome do arquivo permanece compatível)
    df_csv = pd.DataFrame(
        {
            "od": top_ods,
            "n_viagens_validacao": val_tot.reindex(top_ods).fillna(0).astype(int).values,
            "n_viagens_sintetico": syn_tot.reindex(top_ods).fillna(0).astype(int).values,
            "participacao_validacao_pct": (val_den.reindex(top_ods).fillna(0.0).values * 100.0),
            "participacao_sintetico_pct": (syn_den.reindex(top_ods).fillna(0.0).values * 100.0),
        }
    )
    out_csv = GRAFICOS_DIR / f"od_top{int(top_n)}_participacao_validacao_vs_sintetico.csv"
    df_csv.to_csv(out_csv, index=False)
    print(f"[OK] CSV salvo: {out_csv}")
    return top_ods


def plot_horario_por_od(val: pd.DataFrame, syn: pd.DataFrame, ods: list[str], top_n: int = TOP_N) -> None:
    hours = np.arange(24)
    rows: list[dict] = []
    hour_dir = GRAFICOS_DIR / f"od_top{int(top_n)}_por_hora"
    for od in ods:
        v = (
            val[val["od"] == od]
            .groupby("hora_do_dia")["n_viagens"].sum()
            .reindex(hours, fill_value=0)
        )
        s = (
            syn[syn["od"] == od]
            .groupby("hora_do_dia")["n_viagens"].sum()
            .reindex(hours, fill_value=0)
        )
        v_den = v / max(v.sum(), 1)
        s_den = s / max(s.sum(), 1)

        x = np.arange(24)
        width = 0.42
        fig, ax = plt.subplots(figsize=(10, 3.6))
        r1 = ax.bar(x - width / 2, v_den.values, width=width, label="Validação")
        r2 = ax.bar(x + width / 2, s_den.values, width=width, label="Sintético")
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in range(24)])
        ax.set_xlabel("Hora do dia")
        ax.set_ylabel("Proporção (%)")
        ax.set_title(od)
        ax.legend()
        # Espaço para rótulos acima das barras
        ymax = float(max(v_den.max(), s_den.max())) if len(v_den) else 0.0
        ax.set_ylim(0, min(1.05, ymax + 0.06))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        _annotate_bars(ax, r1)
        _annotate_bars(ax, r2)
        fig.tight_layout()

        fname = f"od_{_slugify(od)}_top{int(top_n)}_distrib_hora_validacao_vs_sintetico.png"
        out = hour_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[OK] Gráfico salvo: {out}")

        # Acumula linhas para CSV consolidado por hora
        rows.extend(
            {
                "od": od,
                "hora_do_dia": int(h),
                "validacao_pct": float(v_den.iloc[h] * 100.0),
                "sintetico_pct": float(s_den.iloc[h] * 100.0),
            }
            for h in hours
        )

    # Salva CSV consolidado de distribuições horárias das Top 15
    if rows:
        df_out = pd.DataFrame(rows)
        out_csv = hour_dir / f"od_top{int(top_n)}_distrib_hora_validacao_vs_sintetico.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"[OK] CSV salvo: {out_csv}")


def main() -> None:
    _ensure_dirs(TOP_N)
    val, syn = _load_inputs()
    top_ods = plot_participacao_topN(val, syn, TOP_N)
    if top_ods:
        plot_horario_por_od(val, syn, top_ods, TOP_N)


if __name__ == "__main__":
    main()
