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
from models import DLinearModel, train_model, optimize_dlinear
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            synth_data = synth_data.sort_values("tpep_pickup_datetime").reset_index(drop=True)
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
            for typ in ["real", "synthetic", "real+synthetic"]:
                print(f"\n--- Buscando hiperparâmetros para dados: '{typ}' ---")

                X_train = processed_data[typ]['X_train']
                y_train = processed_data[typ]['y_train']

                seq_len = X_train.shape[1]
                input_dim = X_train.shape[2]
                output_dim = y_train.shape[2]

                study = optimize_dlinear(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    seq_len=seq_len,
                    n_trials=30,
                    seed=seed,
                )

                best = study.best_trials[0]
                params = best.params

                model = (
                    DLinearModel(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        seq_len=seq_len,
                    ).to(device)
                )


                history = train_model(
                    model=model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    epochs=params["epochs"],
                    learning_rate=params["learning_rate"],
                    batch_size=params["batch_size"]
                    #device=device,
                )

                with torch.no_grad():
                    preds = model(X_val.to(device)).cpu().numpy().flatten()
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
                print(
                    f"  > Melhor trial: {params} | Val Loss: {final_val_loss:.6f}"
                )

            if all(len(metrics_results[lbl]) == len(metric_names) for lbl in metrics_results):
                suptitle = f"Comparação de Desempenho\nseed={seed}"
                out_path = os.path.join(
                    "arquivos/d_linear_modulos/save_data/",
                    f"comparacao_optuna_seed_{seed}.png",
                )

                boxplot_model_eval(
                    metrics_dict=metrics_results,
                    metric_names=metric_names,
                    suptitle=suptitle,
                    save_path=out_path,
                )
                print(f"> Figura salva em: {out_path}")

                metrics_results = {k: {} for k in metrics_results}
                    
print("\n--- Treinamento e experimentação concluídos! ---")

if __name__ == "__main__":
    main()
