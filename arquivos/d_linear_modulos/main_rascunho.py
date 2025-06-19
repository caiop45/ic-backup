import torch, numpy as np, pandas as pd
import os
from config import (
    INPUT_WINDOW,
    SYNTHETIC_MULTIPLIER,
    NUM_RUNS,
    DATE_SAMPLER_SEED,
    SAVE_DIR,
)
from data_processing.loader            import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer      import scale_features
from data_processing.dlinear_preparer  import (
    prepare_dlinear_tensors,
    build_input_target_pairs,
    apply_growth_weighting,
    prepare_and_group_datasets,
)
from synthetic_data.date_sampler       import make_date_sampler
from synthetic_data.generator          import (
    gen_synth_data,
    equal_freq,
    perturb_counts,  #  qmap/jitter ficam opcionais
)
from models.gmm_model import multiple_optuna_runs
from models import DLinearModel, train_model, optimize_dlinear
from evaluation.metrics                import compute_metrics
from evaluation.plotting               import generate_plots, plot_hourly_trip_comparison, plot_random_pair_heatmaps, boxplot_model_eval 

from pycave.bayes import GaussianMixture
from utils.helpers import decode_hour_from_sincos, group_trips_by_zone, smape, set_global_seed
from utils.zone_id import assign_zone_names, filter_by_zone
import matplotlib.pyplot as plt      
from sklearn.metrics import r2_score, mean_absolute_error 
# ──────────────────────────────────────────────────────────────
def main() -> None:
    BASE_SEED = 42       
    set_global_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_zones = [150, 151, 152]
    
    # Upload os dados de viagem
    (
        dados_reais_orig,        
        _gmm_placeholder,        
        dados_reaisynth_dataear_input,
        hour_counts_dict_real,    
        GMM_FEATURES,             
    ) = load_real_data()

    #
    ## Split temporal 60 / 20 / 20 

    
    #  GAussian Mixture Model
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    dados_reais_gmm_train = filter_by_zone(df = gmm_train[GMM_FEATURES].astype(np.float32), pu_id = target_zones)
    #breakpoint()

    # Dlinear
    dados_reais_temporal_model_train, dados_reais_temporal_model_val, dados_reais_temporal_model_hold = split_dataset_weekly(
        dados_reaisynth_dataear_input,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )

    # Scaler GMM data
    gmm_scaler, X_scaled = scale_features(dados_reais_gmm_train)

    # Sample-função de datas
    sample_date = make_date_sampler(gmm_train, seed=DATE_SAMPLER_SEED)

   # breakpoint()
    gmm, best_params, best_bic, best_aic = multiple_optuna_runs(
    X_scaled,
    search_space= {
        "n_components": (1, 50),  
        "cov_reg": (1e-7, 1e-3),
        "cov_type": ["full", "diag"],
    },
    seeds = np.random.default_rng(BASE_SEED).integers(
        low=0,              # inclusivo
        high=2**31 - 1,     # exclusivo (portanto vai até 2**31-2)
        size=NUM_RUNS
    ).tolist(),    
    n_trials=500
    )
    gmm.fit(X_scaled)

    gmm.save(r"arquivos/d_linear_modulos/models")
   
    n_synth = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)

    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")
    
    dados_reais_gmm_train["hour_of_day"] = decode_hour_from_sincos(
        dados_reais_gmm_train["sin_hr"], dados_reais_gmm_train["cos_hr"]
    )
    dados_reais_gmm_train["trip_count"] = 1
    dados_reais_gmm_train = assign_zone_names(dados_reais_gmm_train)
    dados_reais_temporal_model_train = assign_zone_names(dados_reais_temporal_model_train, pu_id= target_zones)
    dados_reais_temporal_model_val = assign_zone_names(dados_reais_temporal_model_val, pu_id= target_zones)
    dados_reais_temporal_model_val['tpep_pickup_datetime'] = pd.to_datetime(
        dados_reais_temporal_model_val['tpep_pickup_datetime']
    ).dt.floor('15min')

    for run in range(NUM_RUNS):
            run_seed = BASE_SEED + run   
            set_global_seed(run_seed)

            #Gera os dados sintéticos
            synth_raw_data  = gen_synth_data(gmm, n_synth, gmm_scaler, GMM_FEATURES)
            synth_raw_data = assign_zone_names(synth_raw_data, pu_id= target_zones)
            synth_data = synth_raw_data
            synth_data["tpep_pickup_datetime"] = (
            synth_data["hour_of_day"].dt.hour.astype(int).apply(sample_date)
            + pd.to_timedelta(          # converte "HH:MM:SS" em duração
                synth_data["hour_of_day"].dt.strftime("%H:%M:%S")  # 00 s se não existir
            )
              )
            synth_data = synth_data.sort_values("tpep_pickup_datetime").reset_index(drop=True)
            synth_data = synth_data.dropna(subset=["tpep_pickup_datetime"])
           
            (
                hybrid_temporal_model_train_grouped,
                dados_reais_temporal_model_train_grouped,
                synth_data_grouped,
                dados_reais_temporal_model_val,
            ) = prepare_and_group_datasets(
                real_data=dados_reais_temporal_model_train,
                synthetic_data=synth_data,
                eval_real_data=dados_reais_temporal_model_val,
                only_existing_zones= True
            )
            print(hybrid_temporal_model_train_grouped.head())

            #sbreakpoint()
            groups = {
                "real"          : dados_reais_temporal_model_train_grouped,
                "synthetic"     : synth_data_grouped,
                "real+synthetic": hybrid_temporal_model_train_grouped,
            }
            #breakpoint()
            processed_data = prepare_dlinear_tensors(
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
            #breakpoint()
            metric_names = ["R²", "SMAPE", "MAE"]
            for typ in ["synthetic", "real", "real+synthetic"]:
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
                    n_trials=20,
                    seed=run_seed,
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
                suptitle = f"Comparação de Desempenho\nseed={run_seed}"
                out_path = os.path.join(
                    "arquivos/d_linear_modulos/save_data/",
                    f"comparacao_optuna_seed_{run_seed}.png",
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
