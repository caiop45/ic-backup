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
from evaluation.plotting               import generate_plots, plot_hourly_trip_comparison, plot_random_pair_heatmaps
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
    #@Upload Taxi Trip data
    (
        dados_reais_orig,        
        _gmm_placeholder,        
        dados_reaisynth_dataear_input,
        hour_counts_dict_real,    
        GMM_FEATURES,             
    ) = load_real_data()

    # ───────────────────────────────────────────────────────────────────────────
    # 1a) Split temporal 60 / 20 / 20 para **todos** os conjuntos --------------
    # -------------------------  G M M  ----------------------------------------
    
    #@ Prepare GMM datasets, splitting the data into complete weeks
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    dados_reais_gmm_train       = gmm_train[GMM_FEATURES].astype(np.float32)
    # ----------------------  D L i n e a r  -----------------------------------

    #@ Prepare D-Linear datasets, splitting the data into complete weeks
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
    #@ Apply StdScaler to data 
    gmm_scaler, X_scaled = scale_features(dados_reais_gmm_train)

    # Sample-função de datas
    #@ Assigns a day to synthetic trips based on the real data's distribution
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

            plot_random_pair_heatmaps(
                real_data=dados_reais_gmm_train,
                synth_data=s_pert,
                run_number=run + 1,
                save_dir="/home/caioloss/arquivos/d_linear_modulos/save_data",
                num_pairs=20,          # Este argumento é opcional
                random_seed=run        # Garante reprodutibilidade em cada run
            )

            # 7) (Opcional) continuar chamando outros plots que você já tenha
            plot_hourly_trip_comparison(
                real_df=dados_reais_gmm_train,
                synth_df=s_pert,
                run_idx=run + 1,
                out_dir="/home/caioloss/arquivos/d_linear_modulos/save_data",
            )

if __name__ == "__main__":
    main()
