import torch, numpy as np, pandas as pd
import os
from joblib import dump, load
from pathlib import Path
from config import (
    INPUT_WINDOW,
    SYNTHETIC_MULTIPLIER,
    NUM_RUNS,
    DATE_SAMPLER_SEED,
    SAVE_DIR,
)
from data_processing.loader              import load_real_data, split_dataset_weekly
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
    perturb_counts,
)
from models.gmm_model import multiple_optuna_runs
from models import DLinearModel, train_model, optimize_dlinear
from evaluation.metrics              import compute_metrics
from evaluation.plotting               import generate_plots, plot_hourly_trip_comparison, plot_random_pair_heatmaps, boxplot_model_eval

from pycave.bayes import GaussianMixture
from utils.helpers import decode_hour_from_sincos, group_trips_by_zone, smape, set_global_seed
from utils.zone_id import assign_zone_names, filter_by_zone
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# ──────────────────────────────────────────────────────────────
def main() -> None:
    # ============================ INÍCIO DO SETUP ============================
    print("[DIAGNÓSTICO] Iniciando a execução da função main().")
    BASE_SEED = 50
    set_global_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DIAGNÓSTICO] Usando dispositivo: {device}")
    target_zones = [151, 150, 152]
    print(f"[DIAGNÓSTICO] Zonas de interesse definidas: {target_zones}")

    # ============================ CARGA DE DADOS ============================
    print("\n[DIAGNÓSTICO] Carregando dados reais...")
    (
        dados_reais_orig,
        _gmm_placeholder,
        dados_reais_dlinear_input,
        hour_counts_dict_real,
        GMM_FEATURES,
        DATASAMPLER_FEATURES
    ) = load_real_data()
    print("[DIAGNÓSTICO] Dados carregados.")
    print(f"[DIAGNÓSTICO] Shape de dados_reais_orig: {dados_reais_orig.shape}")
    print(f"[DIAGNÓSTICO] Shape de dados_reais_dlinear_input: {dados_reais_dlinear_input.shape}")

    # ============================ SPLIT TEMPORAL ============================
    print("\n[DIAGNÓSTICO] Realizando split temporal (60/20/20) dos dados...")
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(gmm_full, train_frac=0.60, val_frac=0.20, datetime_col="tpep_pickup_datetime")
    dados_reais_gmm_train   = filter_by_zone(df=gmm_train[GMM_FEATURES].astype(np.float32), pu_id=target_zones)
    dados_reais_date_sample = filter_by_zone(df=gmm_train[DATASAMPLER_FEATURES],       pu_id=target_zones)
    
    dados_reais_temporal_model_train, dados_reais_temporal_model_val, dados_reais_temporal_model_hold = split_dataset_weekly(
        dados_reais_dlinear_input,
        train_frac=0.60, val_frac=0.20, datetime_col="tpep_pickup_datetime",
    )
    print(f"[DIAGNÓSTICO] Shape do dataset de TREINO para DLinear: {dados_reais_temporal_model_train.shape}")
    print(f"[DIAGNÓSTICO] Shape do dataset de VALIDAÇÃO para DLinear: {dados_reais_temporal_model_val.shape}")

    # ============================ PREPARAÇÃO PARA GERAÇÃO SINTÉTICA ============================
    sample_date = make_date_sampler(dados_reais_date_sample, seed=DATE_SAMPLER_SEED)
    print(f"\n[DIAGNÓSTICO] Date Sampler preparado. Tamanho dos dados para amostragem: {len(dados_reais_date_sample)}")



    gmm_scaler, X_scaled = scale_features(dados_reais_gmm_train)
    gmm, best_params, best_bic, best_aic = multiple_optuna_runs(
    X_scaled,
    feature_names= GMM_FEATURES,
    search_space= {
       "n_components": (1, 10),  
        "cov_reg": (1e-5, 1e-3),
        "cov_type": ["full", "diag"],
    },
    seeds = np.random.default_rng(BASE_SEED).integers(
        low=0,              # inclusivo
       high=2**31 - 1,     # exclusivo (portanto vai até 2**31-2)
        size=NUM_RUNS
    ).tolist(),    
    n_trials=200
    ) 
    gmm.fit(X_scaled)
    dump(gmm_scaler, "arquivos/d_linear_modulos/models/dlinear_params/scaler.pkl")
    gmm.save(r"arquivos/d_linear_modulos/models/dlinear_params")
    
    """  gmm         = GaussianMixture.load(Path("arquivos/d_linear_modulos/models/dlinear_params"))
    gmm_scaler  = load(Path("arquivos/d_linear_modulos/models/dlinear_params/scaler.pkl")) 
  """


    print("[DIAGNÓSTICO] Modelos GMM e Scaler carregados.")

    # Filtra zonas de interesse nos datasets temporais
    dados_reais_temporal_model_train = assign_zone_names(dados_reais_temporal_model_train, pu_id=target_zones)
    print("---")
    dados_reais_temporal_model_val   = assign_zone_names(dados_reais_temporal_model_val, pu_id=target_zones)

    # ============================ LOOP DE EXECUÇÃO (RUNS) ============================
    for run in range(NUM_RUNS):
        run_seed = BASE_SEED + run
        set_global_seed(run_seed)
        print(f"\n\n{'='*30} INICIANDO RUN {run + 1}/{NUM_RUNS} com SEED={run_seed} {'='*30}")

        # ============================ GERAÇÃO DE DADOS SINTÉTICOS ============================
        print("\n[DIAGNÓSTICO] Gerando dados sintéticos...")
        n_synth_samples = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)
        print(f"[DIAGNÓSTICO] Número de amostras sintéticas a serem geradas: {n_synth_samples}")
        synth_raw_data = gen_synth_data(gmm, n_synth_samples, gmm_scaler, GMM_FEATURES, sample_date)
        synth_data = assign_zone_names(synth_raw_data, pu_id=target_zones)
        print(f"[DIAGNÓSTICO] Dados sintéticos gerados. Shape: {synth_data.shape}")
        
        # ============================ AGRUPAMENTO DE DATASETS ============================
        print("\n[DIAGNÓSTICO] Agrupando e preparando datasets (real, sintético, híbrido)...")
        (
            hybrid_temporal_model_train_grouped,
            dados_reais_temporal_model_train_grouped,
            synth_data_grouped,
            dados_reais_temporal_model_val_processed,
        ) = prepare_and_group_datasets(
            real_data=dados_reais_temporal_model_train,
            synthetic_data=synth_data,
            eval_real_data=dados_reais_temporal_model_val.copy(deep = True),
            only_existing_zones=True,
        )

        print(f"[SUM] dados reais (train)     : {dados_reais_temporal_model_train_grouped['Manhattan Valley'].sum()}")
        print(f"[SUM] dados sintéticos        : {synth_data_grouped['Manhattan Valley'].sum()}")
        print(f"[SUM] dados de validação (val): {dados_reais_temporal_model_val_processed['Manhattan Valley'].sum()}")

        print("[DIAGNÓSTICO] Shapes dos dataframes agrupados:")
        print(f"  - Real (treino): {dados_reais_temporal_model_train_grouped.shape}")
        print(f"  - Sintético (treino): {synth_data_grouped.shape}")
        print(f"  - Híbrido (treino): {hybrid_temporal_model_train_grouped.shape}")
        print(f"  - Real (validação): {dados_reais_temporal_model_val_processed.shape}")

        #Salva os dados após normalização

        groups2 = { "real": dados_reais_temporal_model_train_grouped, "synthetic": synth_data_grouped, "real+synthetic": hybrid_temporal_model_train_grouped, "real_data_eval": dados_reais_temporal_model_val_processed }
        save_dir = Path("arquivos/d_linear_modulos/save_data/groups")
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, df in groups2.items():
            out_path = save_dir / f"{name}_group.csv"
            # se não quiser o índice no arquivo, use index=False
            df.to_csv(out_path, index=False)
            print(f"> Grupo '{name}' salvo em: {out_path}")

        groups = { "real": dados_reais_temporal_model_train_grouped, "synthetic": synth_data_grouped, "real+synthetic": hybrid_temporal_model_train_grouped }

        target_column_name = 'Manhattan Valley' 
        
        # 2. Defina as colunas que servirão de features (alvo + features extras)
        features_to_keep = [target_column_name, 'hour_of_day']
        
        print(f"\n[AÇÃO] Filtrando datasets para manter apenas as colunas: {features_to_keep}")

        # 3. Filtre os dataframes para manter apenas essas colunas
        for key in groups:
            # Garante que as colunas existem antes de tentar filtrar
            cols = [col for col in features_to_keep if col in groups[key].columns]
            groups[key] = groups[key][cols]

        # Filtra também o dataframe de validação
        cols_val = [col for col in features_to_keep if col in dados_reais_temporal_model_val_processed.columns]
        dados_reais_temporal_model_val_final = dados_reais_temporal_model_val_processed[cols_val]
        

        print("[DIAGNÓSTICO] Colunas FINAIS para treino 'real':", groups['real'].columns.tolist())
        print("[DIAGNÓSTICO] Colunas FINAIS para validação:", dados_reais_temporal_model_val_final.columns.tolist())
        # --- FIM DA AÇÃO CORRETIVA ---

        # ============================ PREPARAÇÃO DOS TENSORES ============================
        print("\n[DIAGNÓSTICO] Preparando tensores para o DLinear...")
          # 1. Inspecionando o DataFrame de TREINO 'REAL'
        print("\n--- INSPECIONANDO O DATASET DE TREINO 'REAL' (groups['real']) ---")
        df_train_real = groups['real']
        print("\n[DEBUG] Informações Gerais (df.info()):")
        df_train_real.info()
        print("\n[DEBUG] Primeiras 5 linhas (df.head()):")
        print(df_train_real.head())
        print("\n[DEBUG] Últimas 5 linhas (df.tail()):")
        print(df_train_real.tail())
        print("\n[DEBUG] Estatísticas Descritivas (df.describe()):")
        print(df_train_real.describe())

        # 2. Inspecionando o DataFrame de VALIDAÇÃO
        print("\n\n--- INSPECIONANDO O DATASET DE VALIDAÇÃO (dados_reais_temporal_model_val_final) ---")
        df_validation = dados_reais_temporal_model_val_final
        print("\n[DEBUG] Informações Gerais (df.info()):")
        df_validation.info()
        print("\n[DEBUG] Primeiras 5 linhas (df.head()):")
        print(df_validation.head())
        print("\n[DEBUG] Últimas 5 linhas (df.tail()):")
        print(df_validation.tail())
        print("\n[DEBUG] Estatísticas Descritivas (df.describe()):")
        print(df_validation.describe())

        print("\n" + "="*27 + " FIM DA INSPEÇÃO DE DEBUG " + "="*27 + "\n\n")
        processed_data = prepare_dlinear_tensors(
            training_groups=groups,
            validation_df=dados_reais_temporal_model_val_final,
            input_window_size=4,
            prediction_horizon=1,
        )
        print("[DIAGNÓSTICO] Detalhes dos tensores criados:")
        for group_name, data_dict in processed_data.items():
            print(f"  - Grupo: '{group_name}'")
            for tensor_name, tensor_data in data_dict.items():
                print(f"    - {tensor_name}: shape={tensor_data.shape}, dtype={tensor_data.dtype}")

        X_val = processed_data["validation"]["X_val"]
        y_val = processed_data["validation"]["y_val"]
        
        # ============================ LOOP DE TREINO E EXPERIMENTAÇÃO ============================
        metrics_results = {"Real": {}, "Sintético": {}, "Real + Sintético":{}}
        metric_names = ["R²", "SMAPE", "MAE"]

        for typ in ["real", "synthetic", "real+synthetic"]:
            print(f"\n\n--- Processando tipo de dados: '{typ}' ---")

            X_train = processed_data[typ]["X_train"]
            y_train = processed_data[typ]["y_train"]

            seq_len   = X_train.shape[1]
            input_dim = X_train.shape[2]
            output_dim = y_train.shape[2]
            
            print(f"[DIAGNÓSTICO] Parâmetros para o modelo '{typ}':")
            print(f"  - Shape X_train: {X_train.shape}")
            print(f"  - Shape y_train: {y_train.shape}")
            print(f"  - Shape X_val:   {X_val.shape}")
            print(f"  - Shape y_val:   {y_val.shape}")
            print(f"  - seq_len:   {seq_len}")
            print(f"  - input_dim: {input_dim}")
            print(f"  - output_dim: {output_dim} <--- IMPORTANTE: Este é o número de alvos que o modelo irá prever.")


            study = optimize_dlinear(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                n_trials=50,
                seed=run_seed,
            ) 

            best   = study.best_trials[0]
            params = best.params 
         
            """ params = {
                "learning_rate" : 4.757299844092417e-05,
                "batch_size" : 8,
                "epochs" : 300,
            }  """
            print(f"[DIAGNÓSTICO] Parâmetros: {params}")

            model = DLinearModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len).to(device)
            
            print(f"\n[DIAGNÓSTICO] Treinando o modelo para '{typ}'...")
            history = train_model(
                model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                epochs=params["epochs"], learning_rate=params["learning_rate"],
                batch_size=params["batch_size"], patience=10, min_delta=0.5,
                restore_best_weights=True,
            )
            print("[DIAGNÓSTICO] Treinamento concluído.")

            # ============================ BLOCO DE PREDIÇÃO E SALVAMENTO (CORRIGIDO E COM DIAGNÓSTICOS) ============================
            print("\n[DIAGNÓSTICO] Iniciando predição no conjunto de validação...")
            with torch.no_grad():
                # Obter predições, mas MANTER a estrutura original (não achatar/flatten)
                preds_tensor = model(X_val.to(device)).cpu()
            
            print(f"[DIAGNÓSTICO] Predição concluída. Shape do tensor de predições (preds_tensor): {preds_tensor.shape}")
            print(f"[DIAGNÓSTICO] Shape do tensor de valores reais (y_val): {y_val.shape}")
            
            save_predictions_dir = Path("arquivos/d_linear_modulos/save_data/predictions")
            save_predictions_dir.mkdir(parents=True, exist_ok=True)

            prediction_results = []
            num_samples = X_val.shape[0]
            print(f"[DIAGNÓSTICO] Preparando para salvar {num_samples} amostras em CSV.")

            for i in range(num_samples):
                input_window = X_val[i].cpu().numpy()
                predictions_for_sample = preds_tensor[i].numpy().flatten()
                true_values_for_sample = y_val[i].cpu().numpy().flatten()

                # Print de diagnóstico para as 3 primeiras amostras
                if i < 3:
                    print(f"\n  --- Amostra de Validação i={i} ---")
                    print(f"  Shape do input_window: {input_window.shape}")
                    print(f"  Shape de predictions_for_sample (após flatten): {predictions_for_sample.shape}")
                    print(f"  Conteúdo de predictions_for_sample: {predictions_for_sample}")
                    print(f"  Shape de true_values_for_sample (após flatten): {true_values_for_sample.shape}")
                    print(f"  Conteúdo de true_values_for_sample: {true_values_for_sample}")

                for target_idx in range(len(predictions_for_sample)):
                    prediction_results.append({
                        "input_window": str(input_window.tolist()),
                        "target_id": target_idx,
                        "predicted_value": predictions_for_sample[target_idx],
                        "real_value": true_values_for_sample[target_idx]
                    })

            df_predictions = pd.DataFrame(prediction_results)
            print(f"\n[DIAGNÓSTICO] DataFrame de predições criado com {len(df_predictions)} linhas.")
            print("[DIAGNÓSTICO] Head do DataFrame de predições:")
            print(df_predictions.head())
            
            predictions_csv_path = save_predictions_dir / f"predictions_{typ}_seed_{run_seed}.csv"
            df_predictions.to_csv(predictions_csv_path, index=False, encoding='utf-8')
            print(f"[DIAGNÓSTICO] Dados de predição para '{typ}' salvos em: {predictions_csv_path}")

            # Para o cálculo de métricas, usamos os valores totais achatados
            preds_for_metrics = preds_tensor.numpy().flatten()
            true_vals_for_metrics = y_val.cpu().numpy().flatten()
            print(f"[DIAGNÓSTICO] Shape de preds_for_metrics (para cálculo de métricas): {preds_for_metrics.shape}")
            print(f"[DIAGNÓSTICO] Shape de true_vals_for_metrics (para cálculo de métricas): {true_vals_for_metrics.shape}")
            
            # ============================ CÁLCULO DE MÉTRICAS ============================
            print("\n[DIAGNÓSTICO] Calculando métricas de desempenho...")
            label_map = {"real": "Real", "synthetic": "Sintético", "real+synthetic": "Real + Sintético"}
            lbl = label_map[typ]
            metrics_results[lbl]["R²"]    = r2_score(true_vals_for_metrics, preds_for_metrics)
            metrics_results[lbl]["SMAPE"] = smape(true_vals_for_metrics, preds_for_metrics)
            metrics_results[lbl]["MAE"]   = mean_absolute_error(true_vals_for_metrics, preds_for_metrics)
            final_val_loss = history["val_loss"][-1]
            print(f"  > Parâmetros: {params} | Val Loss: {final_val_loss:.6f}")
            print(f"  > Métricas para '{lbl}': R²={metrics_results[lbl]['R²']:.4f}, SMAPE={metrics_results[lbl]['SMAPE']:.4f}, MAE={metrics_results[lbl]['MAE']:.4f}")

        # ============================ GERAÇÃO DE GRÁFICOS ============================
        print("\n[DIAGNÓSTICO] Verificando se é possível gerar o gráfico de comparação...")
        if all(metrics_results.get(lbl) for lbl in ["Real", "Sintético", "Real + Sintético"]):
            suptitle = f"Comparação de Desempenho\nseed={run_seed}"
            out_path = os.path.join("arquivos/d_linear_modulos/save_data/", f"comparacao_modelos_seed_{run_seed}.png")
            print(f"[DIAGNÓSTICO] Gerando gráfico de comparação em: {out_path}")
            boxplot_model_eval(
                metrics_dict=metrics_results,
                metric_names=metric_names,
                suptitle=suptitle,
                save_path=out_path,
            )
            print(f"> Figura salva em: {out_path}")
        else:
            print("[DIAGNÓSTICO] Gráfico não gerado. Faltam resultados de um ou mais tipos de modelo.")

    print("\n\n[DIAGNÓSTICO] --- Treinamento e experimentação concluídos! ---")

if __name__ == "__main__":
    main()