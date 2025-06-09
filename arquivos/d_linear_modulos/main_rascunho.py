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
from data_processing.dlinear_preparer  import prepare_all_data_for_dlinear, build_pairs_df, apply_growth_weighting, preparar_e_agrupar_datasets
from synthetic_data.date_sampler       import make_date_sampler
from synthetic_data.generator          import (
    synth_samples_cod1,
    equal_freq,
    perturb_counts,  #  qmap/jitter ficam opcionais
    adjust_counts_by_group
)
from models.gmm_model import multiple_optuna_runs
from models.dlinear                    import DLinearModel, train_model
from evaluation.metrics                import compute_metrics
from evaluation.plotting               import generate_plots, plot_hourly_trip_comparison, plot_random_pair_heatmaps, boxplot_model_eval 
from synthetic_data.min_trips import (
    get_min_daily_trips,
    downsample_to_min_daily,
)
from pycave.bayes import GaussianMixture
from utils.helpers import decode_hour, agrupar_viagens_por_local, smape
from utils.zone_id import add_location_ids_cupy
import matplotlib.pyplot as plt      
from sklearn.metrics import r2_score, mean_absolute_error 
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
    dados_reais_temporal_model_train = dlin_train
    dados_reais_temporal_model_val   = dlin_val
    dados_reais_temporal_model_hold  = dlin_hold
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
    dados_reais_temporal_model_train = add_location_ids_cupy(dados_reais_temporal_model_train)
    dados_reais_temporal_model_val= add_location_ids_cupy(dados_reais_temporal_model_val)
    dados_reais_temporal_model_val['tpep_pickup_datetime'] = pd.to_datetime(
        dados_reais_temporal_model_val['tpep_pickup_datetime']
    ).dt.floor('15min')
    #dados_reais_temporal_model_val = agrupar_viagens_por_local(dados_reais_temporal_model_val)
    for run in range(NUM_EXECUCOES):
            #Gera os dados sintéticos
            synth_raw_data  = synth_samples_cod1(gmm, n_synth, gmm_scaler, GMM_FEATURES)
            synth_raw_data = add_location_ids_cupy(synth_raw_data)
            synth_data = synth_raw_data
            minute_offsets = rng.integers(0, 4, size=len(synth_data)) * 15
            synth_data["tpep_pickup_datetime"] = (
                synth_data["hora_do_dia"].apply(sample_date)
                + pd.to_timedelta(synth_data["hora_do_dia"], unit="h")
                + pd.to_timedelta(minute_offsets, unit="m")
            )
            synth_data = synth_data.dropna(subset=["tpep_pickup_datetime"])


            hybrid_temporal_model_train_grouped, dados_reais_temporal_model_train_grouped, synth_data_grouped, dados_reais_temporal_model_val = preparar_e_agrupar_datasets(dados_reais = dados_reais_temporal_model_train,
                                                              dados_sinteticos = synth_data, dados_reais_eval= dados_reais_temporal_model_val)

            #breakpoint()
            groups = {
                "real"          : dados_reais_temporal_model_train_grouped,
                "synthetic"     : synth_data_grouped,
                "real+synthetic": hybrid_temporal_model_train_grouped,
            }

            processed_data = prepare_all_data_for_dlinear(
            training_groups=groups,
            validation_df=dados_reais_temporal_model_val,
            input_window_size=8,
            prediction_horizon=1,
            )

            X_val = processed_data['validation']['X_val']
            y_val = processed_data['validation']['y_val']

            # ===================================================================
            # 3. LOOP DE TREINAMENTO E EXPERIMENTAÇÃO
            # ===================================================================
            metrics_results = {
                "Real":            {},
                "Sintético":       {},
                "Real + Sintético":{}
            }
            metric_names = ["R²", "SMAPE", "MAE"]
            for epochs in [50, 80, 100, 150, 200, 250]:
                # Loop sobre os diferentes tipos de dados de treino
                for typ in ["real", "synthetic", "real+synthetic"]:
                    
                    print(f"\n--- INICIANDO TREINO: {epochs} épocas | Dados: '{typ}' ---")
                    
                    # Pega os tensores de treino JÁ PRONTOS do dicionário
                    X_train = processed_data[typ]['X_train']
                    y_train = processed_data[typ]['y_train']
                    
                    seq_len = X_train.shape[1]
                    input_dim = X_train.shape[2]
                    output_dim = y_train.shape[2]
                    # 1. Instancia o modelo
                    dlinear_model = DLinearModel(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        seq_len=seq_len
                    ) 
                    
                    # 2. Chama a função de treino modularizada
                    history = train_model(
                        model=dlinear_model,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        epochs=epochs
                    )

                    with torch.no_grad():
                        preds = dlinear_model(X_val).cpu().numpy().flatten()
                    true_vals = y_val.cpu().numpy().flatten()

                    label_map = {
                        "real": "Real",
                        "synthetic": "Sintético",
                        "real+synthetic": "Real + Sintético",
                    }

                    lbl = label_map[typ]
                    metrics_results[lbl]["R²"]    = r2_score(true_vals, preds)
                    metrics_results[lbl]["SMAPE"] = smape(true_vals, preds)
                    metrics_results[lbl]["MAE"]   = mean_absolute_error(true_vals, preds)
                    final_val_loss = history['val_loss'][-1]
                    print(f"  > Experimento concluído. Perda final na validação: {final_val_loss:.6f}")

                #Plota o boxplot    
                if all(len(metrics_results[lbl]) == len(metric_names) for lbl in metrics_results):
                    suptitle = (
                        f"Comparação de Desempenho\n"
                        f"lr={0.001} | epochs={epochs} | seed={seed}"
                    )
                    out_path = os.path.join(
                        "arquivos/d_linear_modulos/save_data/",
                        f"comparacao_lr_{0.001}_epochs_{epochs}_seed_{seed}.png",
                    )

                    boxplot_model_eval(
                        metrics_dict=metrics_results,
                        metric_names=metric_names,
                        suptitle=suptitle,
                        save_path=out_path,
                    )
                    print(f"> Figura salva em: {out_path}")

                    # Limpa para a próxima iteração de epochs
                    metrics_results = {k: {} for k in metrics_results}    
                    
print("\n--- Treinamento e experimentação concluídos! ---")

if __name__ == "__main__":
    main()
