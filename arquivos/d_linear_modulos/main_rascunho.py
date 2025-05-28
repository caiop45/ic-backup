# main.py
# ──────────────────────────────────────────────────────────────
# Pipeline completo para Geração de Dados Sintéticos (GMM) +
# Treinamento do DLinear + Avaliação.
# Agora TODO dado é dividido em 60 % treino / 20 % validação / 20 % hold-out.
# ──────────────────────────────────────────────────────────────

import torch, numpy as np, pandas as pd

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
)
from models.gmm_model import multiple_optuna_runs
from models.dlinear                    import DLinear, train_model
from evaluation.metrics                import compute_metrics
from evaluation.plotting               import generate_plots
from synthetic_data.min_trips import (
    get_min_daily_trips,
    downsample_to_min_daily,
)
from pycave.bayes import GaussianMixture
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

    #gmm, best_params, best_bic = multiple_optuna_runs(
    #X_scaled,
    #search_space=search_space,
    #seeds = rng.integers(low=0, high=2**31-1, size=NUM_EXECUCOES).tolist(),
    #n_trials=200,
    #save_dir=r"arquivos/d_linear_modulos/models"
    #)

    #Fit manual usando os hiperparâmetros do JSON
    gmm = GaussianMixture(
            num_components=46,
            covariance_type='diag',
            covariance_regularization=0.0009980274958510734,
            trainer_params={"max_epochs": 200,
                            "accelerator": "auto",
                            "devices": 1},
        )
    
    gmm.fit(X_scaled)

    n_synth = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)
    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")

    for run in range(NUM_EXECUCOES):
            #Aqui gera os dados sintéticos e faz um monte de manipulação. Precisa verificar se tá fazendo sentido
            #Usar as métricas que o professor deixou no notion
            s_raw  = synth_samples_cod1(gmm, n_synth, gmm_scaler, GMM_FEATURES)
            s_eq   = equal_freq(s_raw, hour_counts_dict_real, rng)
            s_pert = perturb_counts(s_eq, rng)          
            s_pert["tpep_pickup_datetime"] = (
                s_pert["hora_do_dia"].apply(sample_date) +
                pd.to_timedelta(s_pert["hora_do_dia"], unit="h")
            )
            s_pert = s_pert.dropna(subset=["tpep_pickup_datetime"])

            #Pego o número mínimo de viagens por dia e boto esse limite para cada dia
            min_trips   = get_min_daily_trips(s_pert[["tpep_pickup_datetime", "num_viagens"]])
            dataset_min_trips      = downsample_to_min_daily(s_pert, min_trips, rng)


            #Desconsiderar por enquanto, é necessário apenas para a parte de treinar o dlinear
            #synth_data = s_pert[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()

            #real_plus_synth = pd.concat([dados_reaisynth_dataear_input, synth_data], ignore_index=True)
            #groups = {
            #    "real"          : dados_reaisynth_dataear_input,
            #   "synthetic"     : synth_data,
            #    "real+synthetic": real_plus_synth,
            # }
            #prepared = {k: _prep(v) for k, v in groups.items()}
            #pairs    = {k: build_pairs_df(v) for k, v in prepared.items()}



if __name__ == "__main__":
    main()
