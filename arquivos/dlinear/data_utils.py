import pandas as pd
import numpy as np
import torch
import config # Importa as configurações

def load_and_prepare_initial_data(file_path):
    """Carrega os dados brutos e faz o pré-processamento inicial."""
    try:
        dados = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo parquet não encontrado em {file_path}")
        raise
    dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
    dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
    dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek
    # Filtra apenas Segunda (0), Terça (1), Quarta (2) - conforme original
    dados = dados[dados['dia_da_semana'].between(0, 2)].copy() # Use .copy() para evitar SettingWithCopyWarning
    dados['data_do_dia'] = dados['tpep_pickup_datetime'].dt.date
    return dados

def prepare_gmm_data(df):
    """Seleciona features para o GMM, remove NaNs e converte data para timestamp."""
    dados_gmm = df[config.FEATURES_GMM].dropna().sample(frac=1.0)
    # Converte data para timestamp para o GMM
    dados_gmm['data_do_dia'] = pd.to_datetime(dados_gmm['data_do_dia']).apply(lambda x: x.timestamp())
    return dados_gmm

def process_aggregated_data(df, data_col='data_do_dia', time_col='hora_do_dia', count_col='num_viagens'):
    """Agrupa dados por data/hora, calcula contagem e ordena por datetime."""
    df_grouped = df.groupby([data_col, time_col]).size().reset_index(name=count_col)
    # Certifique-se de que 'data_do_dia' seja datetime antes de adicionar timedelta
    df_grouped[data_col] = pd.to_datetime(df_grouped[data_col])
    df_grouped['datetime'] = df_grouped[data_col] + pd.to_timedelta(df_grouped[time_col], unit='h')
    df_grouped = df_grouped.sort_values('datetime').reset_index(drop=True)
    return df_grouped

def create_sequence_dataset(series_values, window_size=config.WINDOW_SIZE):
    """Cria sequências de dados para modelos de série temporal."""
    X, y = [], []
    # Garante que series_values seja um array numpy para indexação
    series_values = np.asarray(series_values)
    for i in range(len(series_values) - window_size):
        X.append(series_values[i : i + window_size])
        y.append(series_values[i + window_size])
    return np.array(X), np.array(y)

def prepare_tensors(X_np, y_np):
    """Converte arrays NumPy em tensores PyTorch."""
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32).unsqueeze(-1) # Adiciona dimensão para o target
    return X_t, y_t

def split_data(X, y, ratio=config.TRAIN_SPLIT_RATIO):
    """Divide os dados em conjuntos de treino e validação."""
    split_idx = int(ratio * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_val, y_val