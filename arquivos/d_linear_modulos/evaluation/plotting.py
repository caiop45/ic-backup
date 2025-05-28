import math
import seaborn as sns
import matplotlib.pyplot as plt
from config import SAVE_DIR

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
