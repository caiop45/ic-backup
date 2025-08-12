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
    perturb_counts,
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
    BASE_SEED = 372
    set_global_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_zones = [151, 150, 152]  # zonas de interesse

    # Upload dos dados de viagem
    (
        dados_reais_orig,
        _gmm_placeholder,
        dados_reaisynth_dataear_input,
        hour_counts_dict_real,
        GMM_FEATURES,
        DATASAMPLER_FEATURES
    ) = load_real_data()

    # Split temporal 60 / 20 / 20 para o GMM
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    dados_reais_gmm_train   = filter_by_zone(df=gmm_train[GMM_FEATURES].astype(np.float32), pu_id=target_zones)
    dados_reais_date_sample = filter_by_zone(df=gmm_train[DATASAMPLER_FEATURES],       pu_id=target_zones)

    # Split temporal 60 / 20 / 20 para o DLinear
    dados_reais_temporal_model_train, dados_reais_temporal_model_val, dados_reais_temporal_model_hold = split_dataset_weekly(
        dados_reaisynth_dataear_input,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )

    # Criador de datas para o gerador sintético
    sample_date = make_date_sampler(dados_reais_date_sample, seed=DATE_SAMPLER_SEED)

    # Carrega modelos previamente treinados
    gmm         = GaussianMixture.load(Path("arquivos/d_linear_modulos/models/dlinear_params"))
    gmm_scaler  = load(Path("arquivos/d_linear_modulos/models/dlinear_params/scaler.pkl"))

    # Filtra zonas de interesse nos datasets temporais
    dados_reais_temporal_model_train = assign_zone_names(dados_reais_temporal_model_train, pu_id=target_zones)
    dados_reais_temporal_model_val   = assign_zone_names(dados_reais_temporal_model_val,   pu_id=target_zones)

    for run in range(NUM_RUNS):
        run_seed = BASE_SEED + run
        set_global_seed(run_seed)

        # Gera dados sintéticos
        synth_raw_data = gen_synth_data(
            gmm,
            int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER),
            gmm_scaler,
            GMM_FEATURES,
            sample_date,
        )
        synth_data = assign_zone_names(synth_raw_data, pu_id=target_zones)

        # Agrupa e prepara datasets
        (
            hybrid_temporal_model_train_grouped,
            dados_reais_temporal_model_train_grouped,
            synth_data_grouped,
            dados_reais_temporal_model_val,
        ) = prepare_and_group_datasets(
            real_data=dados_reais_temporal_model_train,
            synthetic_data=synth_data,
            eval_real_data=dados_reais_temporal_model_val.copy(deep = True),
            only_existing_zones=True,
        )

        groups = {
            "real":           dados_reais_temporal_model_train_grouped,
            "synthetic":      synth_data_grouped,
            "real+synthetic": hybrid_temporal_model_train_grouped
        }

        save_dir = Path("arquivos/d_linear_modulos/save_data/groups")
        save_dir.mkdir(parents=True, exist_ok=True)

        for name, df in groups.items():
            out_path = save_dir / f"{name}_group.csv"
            # se não quiser o índice no arquivo, use index=False
            df.to_csv(out_path, index=False)
            print(f"> Grupo '{name}' salvo em: {out_path}")

        # ─────────────────────────── DIAGNÓSTICO DOS DATASETS ───────────────────────────
        print("\n================ DIAGNÓSTICO: Datasets antes do DLinear ================")
        for name, df in groups.items():
            print(f"\n[{name.upper()}] linhas={len(df)}, colunas={df.columns.tolist()}")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if not numeric_cols.empty:
                print(df[numeric_cols].describe().loc[["min", "max"]])
            print("Exemplo de linhas:\n", df.head())
        # ────────────────────────────────────────────────────────────────────────────────

        zones_to_exclude = {150, 152}            # ajuste aqui quando quiser
        def drop_zones(df, zones):
            """Remove colunas cujo sufixo (ou o próprio nome) seja o id da zona."""
            # Se as colunas são inteiras (ex.: 150), só faz a interseção direta
            if all(isinstance(c, int) for c in df.columns):
                return df.drop(columns=df.columns.intersection(zones), errors="ignore")

            # Caso venham como strings ou com prefixo/sufixo (ex.: trip_count_150)
            cols_to_drop = [
                c for c in df.columns
                if any(str(z) == str(c) or str(c).endswith(f"_{z}") for z in zones)
            ]
            return df.drop(columns=cols_to_drop, errors="ignore")

        # 1) validação
        dados_reais_temporal_model_val = drop_zones(
            dados_reais_temporal_model_val, zones_to_exclude
        )
        # 2) cada grupo de treino
        for key in ["real", "synthetic", "real+synthetic"]:
            groups[key] = drop_zones(groups[key], zones_to_exclude)

        processed_data = prepare_dlinear_tensors(
            training_groups=groups,
            validation_df=dados_reais_temporal_model_val,
            input_window_size=8,
            prediction_horizon=1,
        )

        # ─────────────────────── DIAGNÓSTICO DOS TENSORES ───────────────────────
        print("\n================ DIAGNÓSTICO: Tensores antes do treino ================")
        for typ in processed_data:
            if typ == "validation":  # pula bloco de validação aqui
                continue
            X_tr = processed_data[typ]["X_train"]
            y_tr = processed_data[typ]["y_train"]
            print(f"[{typ}] X_train shape={tuple(X_tr.shape)}, y_train shape={tuple(y_tr.shape)}")
            print(f"      X_train min={X_tr.min():.3f}, max={X_tr.max():.3f}")
            print(f"      y_train min={y_tr.min():.3f}, max={y_tr.max():.3f}")
        print("=======================================================================\n")
        # ─────────────────────────────────────────────────────────────────────────

        X_val = processed_data["validation"]["X_val"]
        y_val = processed_data["validation"]["y_val"]

        # =======================================================================
        # 3. LOOP DE TREINAMENTO E EXPERIMENTAÇÃO
        # =======================================================================
        metrics_results = {
            "Real":            {},
            "Sintético":       {},
            "Real + Sintético":{}
        }
        metric_names = ["R²", "SMAPE", "MAE"]

        # Para economizar tempo, treina só no conjunto 'real' (ajuste se necessário)
        for typ in ["synthetic", "real", "real+synthetic"]:
            print(f"\n--- Buscando hiperparâmetros para dados: '{typ}' ---")

            X_train = processed_data[typ]["X_train"]
            y_train = processed_data[typ]["y_train"]

            seq_len   = X_train.shape[1]
            input_dim = X_train.shape[2]
            output_dim = y_train.shape[2]

            """ study = optimize_dlinear(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                n_trials=10,
                seed=run_seed,
            ) 

            best   = study.best_trials[0]
             params = best.params
           """
            params = {
            "epochs":        400,
            "learning_rate": 3.396932572663065e-05,
            "batch_size":    1024,
          }

            model = DLinearModel(
                input_dim=input_dim,
                output_dim=output_dim,
                seq_len=seq_len,
            ).to(device)

            history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=params["epochs"],
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                patience=10,
                min_delta=0.5,          # exige melhora ≥ 0.5 no val_loss
                restore_best_weights=True,
            )

            with torch.no_grad():
                preds = model(X_val.to(device)).cpu().numpy().flatten()
            true_vals = y_val.cpu().numpy().flatten()


             # ### INÍCIO DO CÓDIGO ADICIONADO ###
            # -------------------------------------------------------------------
            print(f">>> Preparando dados de input/predição para salvar em CSV...")
            
            # Garante que o diretório de saída exista
            save_predictions_dir = Path("arquivos/d_linear_modulos/save_data/predictions")
            save_predictions_dir.mkdir(parents=True, exist_ok=True)
            
            prediction_results = []
            # Itera sobre cada predição para capturar o input correspondente
            for i in range(len(preds)):
                # Pega a janela de input original (X_val[i]) que gerou a predição i
                input_window_tensor = X_val[i]
                
                # Converte o tensor para uma lista de listas e depois para string
                input_window_str = str(input_window_tensor.cpu().numpy().tolist())
                
                prediction_results.append({
                    "input_window": input_window_str,
                    "predicted_value": preds[i],
                    "real_value": true_vals[i]
                })

            # Cria o DataFrame com os resultados
            df_predictions = pd.DataFrame(prediction_results)
            
            # Define o caminho do arquivo e salva
            predictions_csv_path = save_predictions_dir / f"predictions_{typ}_seed_{run_seed}.csv"
            df_predictions.to_csv(predictions_csv_path, index=False, encoding='utf-8')
            
            print(f">>> Dados de predição para '{typ}' salvos em: {predictions_csv_path}")

            label_map = {
                "real": "Real",
                "synthetic": "Sintético",
                "real+synthetic": "Real + Sintético",
            }

            lbl = label_map[typ]
            metrics_results[lbl]["R²"]    = r2_score(true_vals, preds)
            metrics_results[lbl]["SMAPE"] = smape(true_vals, preds)
            metrics_results[lbl]["MAE"]   = mean_absolute_error(true_vals, preds)
            final_val_loss = history["val_loss"][-1]
            print(f"  > Melhor trial: {params} | Val Loss: {final_val_loss:.6f}")

        # Gera boxplot de comparação
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

    print("\n--- Treinamento e experimentação concluídos! ---")

if __name__ == "__main__":
    main()
