import torch, numpy as np, pandas as pd
import os
from config import (
    WINDOW,
    SYNTHETIC_MULTIPLIER,
    NUM_EXECUCOES,
    DATA_SAMPLER_SEED,
    SAVE_DIR,
)
from data_processing.loader            import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer      import scale_features
from data_processing.dlinear_preparer  import _prep, build_pairs_df, apply_growth_weighting
from synthetic_data.date_sampler       import make_date_sampler
from synthetic_data.generator          import (
    synth_samples_cod1,
    equal_freq,
    perturb_counts,  #  qmap/jitter ficam opcionais
    adjust_counts_by_group
)
from models.gmm_model import multiple_optuna_runs
from models.dlinear                    import DLinear, train_model
from evaluation.metrics                import compute_metrics
from evaluation.plotting               import generate_plots, plot_hourly_trip_comparison
from synthetic_data.min_trips import (
    get_min_daily_trips,
    downsample_to_min_daily,
)
from pycave.bayes import GaussianMixture
from utils.helpers import decode_hour
from utils.zone_id import add_location_ids_cupy
import matplotlib.pyplot as plt       
# ──────────────────────────────────────────────────────────────
def main() -> None:
    # Seeds globais -------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    # ───────────────────────────────────────────────────────────────────────────
    # 1) Carrega dados reais ----------------------------------------------------
    (
        dados_reais_orig,          # DataFrame completo filtrado
        _gmm_placeholder,          # NÃO mais usado para split direto
        dados_reaisynth_dataear_input, # 3 colunas para DLinear
        hour_counts_dict_real,     # {hora: nº de viagens}
        GMM_FEATURES,              # lista das colunas de features
    ) = load_real_data()

    # ───────────────────────────────────────────────────────────────────────────
    # 1a) Split temporal 60 / 20 / 20 para **todos** os conjuntos --------------
    # -------------------------  G M M  ----------------------------------------
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    dados_reais_gmm_train       = gmm_train[GMM_FEATURES].astype(np.float32)
    # ----------------------  D L i n e a r  -----------------------------------
    dlin_train, dlin_val, dlin_hold = split_dataset_weekly(
        dados_reaisynth_dataear_input,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    dados_reaisynth_dataear_train = dlin_train
    dados_reaisynth_dataear_val   = dlin_val
    dados_reaisynth_dataear_hold  = dlin_hold

    if dados_reais_gmm_train.empty or dados_reaisynth_dataear_train.empty:
        raise ValueError("Conjuntos de treino vazios após o pré-processamento!")

      # ----- 2) Escala features p/ GMM -----
    gmm_scaler, X_scaled = scale_features(dados_reais_gmm_train)

    # Sample-função de datas
    sample_date = make_date_sampler(dados_reaisynth_dataear_input, seed=DATA_SAMPLER_SEED)

    #Gerar seeds(Conferir posteriormente se há reprodutibilidade)

    seed = torch.initial_seed() 
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    #Para encontrar os melhores parametros com o optuna

    # Combinações de hiperparametros para o optuna
    search_space = {
        "n_components": (40, 50),  
        "cov_reg": (1e-6, 1e-3),
        "cov_type": ["full", "diag"],
    }

   # gmm, best_params, best_bic = multiple_optuna_runs(
    #X_scaled,
   # search_space=search_space,
   # seeds = rng.integers(low=0, high=2**31-1, size=NUM_EXECUCOES).tolist(),
   # n_trials=200,
   # save_dir=r"arquivos/d_linear_modulos/models"
   # )

    #Fit manual usando os hiperparâmetros do JSON
    gmm = GaussianMixture(
           num_components=46,
           covariance_type='full',
          covariance_regularization=0.0009980274958510734,
          trainer_params={"max_epochs": 200,
                            "accelerator": "auto",
                            "devices": 1},
       )
    
    gmm.fit(X_scaled)

    n_synth = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)
    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")
    dados_reais_gmm_train["hora_do_dia"] = decode_hour(dados_reais_gmm_train["sin_hr"], dados_reais_gmm_train["cos_hr"])
    dados_reais_gmm_train["num_viagens"] = 1 
    dados_reais_gmm_train = add_location_ids_cupy(dados_reais_gmm_train)

    for run in range(NUM_EXECUCOES):
            #Gera os dados sintéticos
            synth_raw_data  = synth_samples_cod1(gmm, n_synth, gmm_scaler, GMM_FEATURES)
            synth_raw_data = add_location_ids_cupy(synth_raw_data)

            #Printa o num de zonas vazias retornadas pela add_Locations_id
            filtros_nulos_ou_vazios = (
            synth_raw_data["PULocationID"].isna() |
            synth_raw_data["DOLocationID"].isna() |
            (synth_raw_data["PULocationID"] == "") |
            (synth_raw_data["DOLocationID"] == "")
            )
            soma_viagens_invalidas = synth_raw_data.loc[filtros_nulos_ou_vazios, "num_viagens"].sum()
           #print(f"\n[SOMA] Total de num_viagens com PULocationID ou DOLocationID nulos ou vazios: {soma_viagens_invalidas}")

            #Precisa verificar essa função aqui. Tem muito dado sintético sendo desconsiderado. 
            #A minha hipótese é de que o GMM tá falhando na hora de gerar os dados espaciais
            #[COMPARAÇÃO] num_viagens total
            #• Sintético ajustado: 689,899
            #• Real             : 1,352,483
            #s_pert = adjust_counts_by_group(s_df= synth_raw_data, r_df = dados_reais_gmm_train, rng = rng)

            s_pert = synth_raw_data
            s_pert["tpep_pickup_datetime"] = (
                s_pert["hora_do_dia"].apply(sample_date) +
                pd.to_timedelta(s_pert["hora_do_dia"], unit="h")
            )
            s_pert = s_pert.dropna(subset=["tpep_pickup_datetime"])

            ##SALVA EM UM CSV OS NUM_VIAGENS SINTÉTICAS E REAIS PARA CADA TRIPLA DISTINTA(PU,DOLOCATIONID E HORA_DO_DIA)
            real_group = (
            dados_reais_gmm_train
            .groupby(["PULocationID", "DOLocationID", "hora_do_dia"])["num_viagens"]
            .sum()
            .reset_index()
            .rename(columns={"num_viagens": "num_viagens_reais"})
        )

            # 2) Agrupar e somar num_viagens nos dados sintéticos (s_pert)
            synth_group = (
                s_pert
                .groupby(["PULocationID", "DOLocationID", "hora_do_dia"])["num_viagens"]
                .sum()
                .reset_index()
                .rename(columns={"num_viagens": "num_viagens_sinteticas"})
            )

            # 3) Fazer merge para ficar todas as combinações, preenchendo NaN com zero
            comparison = (
                pd.merge(
                    real_group,
                    synth_group,
                    on=["PULocationID", "DOLocationID", "hora_do_dia"],
                    how="outer"
                )
                .fillna(0)
            )

            # 4) Salvar em CSV
            csv_path = os.path.join(r"/home/caioloss/arquivos/d_linear_modulos/save_data", f"comparacao_viagens_run_{run+1}.csv")
            comparison.to_csv(csv_path, index=False)

            #Pego o número mínimo de viagens por dia e boto esse limite para cada dia
            #Vai ser usado pra plotar as métricas
            #min_trips   = get_min_daily_trips(s_pert[["tpep_pickup_datetime", "num_viagens"]])
            #dataset_min_trips      = downsample_to_min_daily(s_pert, min_trips, rng)
            
            # ─────────── INÍCIO DA PORÇÃO DE HEATMAPS (20 combinações aleatórias) ────────────

            # 1) Pivot completo dos dados reais e sintéticos (igual ao seu código)
            pivot_real = pd.pivot_table(
                dados_reais_gmm_train,
                index="PULocationID",
                columns="DOLocationID",
                values="num_viagens",
                aggfunc="sum",
                fill_value=0
            )

            pivot_synth = pd.pivot_table(
                s_pert,
                index="PULocationID",
                columns="DOLocationID",
                values="num_viagens",
                aggfunc="sum",
                fill_value=0
            )

            # 2) Selecionar 20 pares (PU, DO) aleatórios a partir dos dados sintéticos
            #    Primeiro, extrai todas as tuplas únicas de (PULocationID, DOLocationID)
            todas_tuplas = s_pert[["PULocationID", "DOLocationID"]].drop_duplicates()
            #    Em seguida, faz a amostragem de 20 dessas tuplas (pode usar random_state para reprodutibilidade)
            sample_pairs = todas_tuplas.sample(n=20, random_state=run)  # use `run` ou outro seed para cada iteração

            # 3) Extrair os conjuntos de PUs e DOs que aparecem nessas 20 tuplas
            pu_ids_selecionados = sample_pairs["PULocationID"].unique().tolist()
            do_ids_selecionados = sample_pairs["DOLocationID"].unique().tolist()

            # 4) “Fatiar” o pivot para manter só os PUs e DOs de interesse
            #    Aqui, usamos reindex para garantir a ordem e preencher zeros caso algum par não exista em um dos datasets
            pivot_real_sub = pivot_real.reindex(
                index=pu_ids_selecionados,
                columns=do_ids_selecionados,
                fill_value=0
            )
            pivot_synth_sub = pivot_synth.reindex(
                index=pu_ids_selecionados,
                columns=do_ids_selecionados,
                fill_value=0
            )

            # 5) Plotar os dois heatmaps em 1×2 subplots, mantendo a mesma escala de cor (vmin/vmax) para
            #    facilitar a comparação
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), constrained_layout=True)

            # 5a) Heatmap (dados reais) — usando o sub-pivot
            im0 = axes[0].imshow(
                pivot_real_sub.values,
                aspect="auto",
                origin="lower",
                interpolation="nearest"
            )
            axes[0].set_title(f"Dados Reais (run {run+1}) — 20 pares aleatórios")
            axes[0].set_xlabel("DOLocationID")
            axes[0].set_ylabel("PULocationID")

            axes[0].set_xticks(range(len(pivot_real_sub.columns)))
            axes[0].set_xticklabels(pivot_real_sub.columns, rotation=90, fontsize=6)
            axes[0].set_yticks(range(len(pivot_real_sub.index)))
            axes[0].set_yticklabels(pivot_real_sub.index, fontsize=6)

            cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar0.set_label("num_viagens (real)")

            # 5b) Heatmap (dados sintéticos) — mesma escala de cor (vmin/vmax)
            vmin = pivot_real_sub.values.min()
            vmax = pivot_real_sub.values.max()

            im1 = axes[1].imshow(
                pivot_synth_sub.values,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax
            )
            axes[1].set_title(f"Dados Sintéticos (run {run+1}) — 20 pares aleatórios")
            axes[1].set_xlabel("DOLocationID")
            axes[1].set_yticks([])  # esconde labels de y, já que o primeiro já mostra

            axes[1].set_xticks(range(len(pivot_synth_sub.columns)))
            axes[1].set_xticklabels(pivot_synth_sub.columns, rotation=90, fontsize=6)

            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            cbar1.set_label("num_viagens (sintético)")

            # 6) Salvar a figura no diretório desejado
            out_dir = "/home/caioloss/arquivos/d_linear_modulos/save_data"
            os.makedirs(out_dir, exist_ok=True)
            fig_path = os.path.join(out_dir, f"heatmap_20pares_run_{run+1}.png")
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)

            # 7) (Opcional) continuar chamando outros plots que você já tenha
            plot_hourly_trip_comparison(
                real_df=dados_reais_gmm_train,
                synth_df=s_pert,
                run_idx=run + 1,
                out_dir=out_dir,
            )

if __name__ == "__main__":
    main()
