import os
import pandas as pd
import numpy as np
import torch
import datetime
import time

# Importa dos módulos locais
import config
import data_utils
import gmm_utils
import models
import evaluation
import plots

def main():
    print("Iniciando o script principal...")
    start_time = time.time()

    # --- 1) Carregamento e Preparação Inicial ---
    print("Carregando e preparando dados iniciais...")
    dados_raw = data_utils.load_and_prepare_initial_data(config.DATA_FILE_PATH)
    dados_gmm = data_utils.prepare_gmm_data(dados_raw)
    print(f"Dados GMM preparados: {dados_gmm.shape[0]} linhas")

    # --- Processamento dos Dados Reais Agregados ---
    print("Processando dados reais agregados...")
    df_real_grouped = data_utils.process_aggregated_data(
        dados_raw[['data_do_dia', 'hora_do_dia']] # Passa apenas as colunas necessárias
    )
    print(f"Dados reais agregados processados: {df_real_grouped.shape[0]} linhas")


    # Dataframe para armazenar todos os resultados
    all_results = []

    # --- Loop Principal de Experimentos ---
    for nc in config.NCS_LIST:
        print(f"\n===== Iniciando experimento para nc = {nc} =====")

        # --- 2) Treinar GMM e Gerar Dados Sintéticos ---
        synthetic_df = gmm_utils.train_gmm_and_generate(dados_gmm, nc)

        # --- 3) Processar Dados Sintéticos Agregados ---
        print("Processando dados sintéticos agregados...")
        df_sint_grouped = data_utils.process_aggregated_data(synthetic_df)
        print(f"Dados sintéticos agregados processados: {df_sint_grouped.shape[0]} linhas")


        # --- 4) Combinar Dados Reais + Sintéticos ---
        print("Combinando dados reais e sintéticos...")
        # Garante que as colunas 'data_do_dia' e 'hora_do_dia' existam e tenham tipos compatíveis
        # Pode ser necessário re-processar o synthetic_df se a agregação anterior o modificou muito
        # Vamos usar o df_sint_grouped que já tem a contagem 'num_viagens'
        df_real_plus_sint_grouped = pd.concat([
            df_real_grouped[['data_do_dia', 'hora_do_dia', 'num_viagens']],
            df_sint_grouped[['data_do_dia', 'hora_do_dia', 'num_viagens']]
        ]).groupby(['data_do_dia', 'hora_do_dia'], as_index=False)['num_viagens'].sum()

        # Recalcula datetime e ordena após a soma
        df_real_plus_sint_grouped['datetime'] = (
            pd.to_datetime(df_real_plus_sint_grouped['data_do_dia'])
            + pd.to_timedelta(df_real_plus_sint_grouped['hora_do_dia'], unit='h')
        )
        df_real_plus_sint_grouped = df_real_plus_sint_grouped.sort_values('datetime').reset_index(drop=True)
        print(f"Dados combinados processados: {df_real_plus_sint_grouped.shape[0]} linhas")


        # --- 5) Criar Sequências ---
        print("Criando sequências...")
        X_real, y_real = data_utils.create_sequence_dataset(df_real_grouped['num_viagens'].values)
        X_sint, y_sint = data_utils.create_sequence_dataset(df_sint_grouped['num_viagens'].values)
        X_real_sint, y_real_sint = data_utils.create_sequence_dataset(df_real_plus_sint_grouped['num_viagens'].values)

        # --- 6) Transformar em Tensores ---
        print("Convertendo para tensores...")
        X_real_t, y_real_t = data_utils.prepare_tensors(X_real, y_real)
        X_sint_t, y_sint_t = data_utils.prepare_tensors(X_sint, y_sint)
        X_real_sint_t, y_real_sint_t = data_utils.prepare_tensors(X_real_sint, y_real_sint)

        datasets = {
            'real': (X_real_t, y_real_t),
            'synthetic': (X_sint_t, y_sint_t),
            'real+synthetic': (X_real_sint_t, y_real_sint_t)
        }

        # --- Loop de Épocas e Seeds ---
        for epochs in config.EPOCHS_LIST:
            print(f"\n--- Iniciando execuções para nc={nc}, epochs={epochs} ---")
            for execucao in range(config.NUM_EXECUCOES):
                seed = torch.initial_seed() + execucao # Gera seeds diferentes para cada execução
                print(f"\nExecução {execucao + 1}/{config.NUM_EXECUCOES} (Seed: {seed})")
                torch.manual_seed(seed)
                np.random.seed(seed) # Define seed para numpy também, se necessário

                for data_type, (X, y) in datasets.items():
                    print(f"  Treinando modelo com dados: {data_type}")
                    if len(X) == 0 or len(y) == 0:
                         print(f"  AVISO: Dados vazios para tipo '{data_type}'. Pulando treinamento.")
                         continue

                    # Define a seed novamente antes de cada treino para reprodutibilidade do DLinear
                    torch.manual_seed(seed)

                    # Dividir dados
                    X_train, y_train, X_val, y_val = data_utils.split_data(X, y)

                    if len(X_train) == 0 or len(X_val) == 0:
                        print(f"  AVISO: Conjunto de treino ou validação vazio para '{data_type}' após split. Pulando.")
                        continue

                    # Instanciar e treinar modelo
                    model = models.DLinear()
                    model = models.train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=config.LEARNING_RATE, verbose=False) # verbose=False para reduzir output

                    # Avaliar modelo
                    model.eval()
                    with torch.no_grad():
                        y_pred_val = model(X_val).cpu().numpy() # Mover para CPU antes de numpy
                    y_true_val = y_val.cpu().numpy()

                    # Calcular métricas
                    metrics = evaluation.calculate_metrics(y_true_val, y_pred_val)
                    print(f"    Métricas ({data_type}): {metrics}")


                    # Salvar resultados
                    for metric, value in metrics.items():
                         if pd.isna(value): # Não salva métricas NaN (ex: R² indefinido)
                             print(f"    AVISO: Métrica '{metric}' é NaN para '{data_type}'. Não será salva.")
                             continue
                         all_results.append({
                            'tipo_dado': data_type,
                            'metrica': metric,
                            'valor': value,
                            'nc': nc,
                            'epochs': epochs,
                            'seed': seed
                        })

    # --- Finalização ---
    print("\n===== Experimentos Concluídos =====")

    # Converter resultados para DataFrame
    df_resultados = pd.DataFrame(all_results)

    # Salvar resultados em CSV
    results_file = os.path.join(config.SAVE_DIR, "resultados_metricas_modular.csv")
    try:
        df_resultados.to_csv(results_file, index=False)
        print(f"Resultados salvos em: {results_file}")
    except Exception as e:
        print(f"Erro ao salvar resultados em CSV: {e}")


    # Gerar gráficos
    if not df_resultados.empty:
         plotting.plot_results(df_resultados, config.SAVE_DIR)
    else:
        print("DataFrame de resultados está vazio. Nenhum gráfico será gerado.")


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tempo total de execução: {datetime.timedelta(seconds=int(total_time))}")
    print("Script principal finalizado.")


# Ponto de entrada do script
if __name__ == "__main__":
    main()