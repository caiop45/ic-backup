import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# --------------------------------------------------
SYNTHETIC_MULTIPLIER = 5
SAVE_DIR = "/home/caioloss/gráficos/linear/"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
dados = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek
dados = dados[dados['dia_da_semana'].between(0, 2)]
dados['data_do_dia'] = dados['tpep_pickup_datetime'].dt.date

features = ['data_do_dia', 'hora_do_dia', 'PU_longitude', 'PU_latitude', 'DO_longitude', 'DO_latitude']
dados_gmm = dados[features].dropna().sample(frac=1.0)
dados_gmm['data_do_dia'] = pd.to_datetime(dados_gmm['data_do_dia']).apply(lambda x: x.toordinal())

# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------
def create_sequence_dataset(series_values, window_size=4):
    X, y = [], []
    for i in range(len(series_values) - window_size):
        X.append(series_values[i:i+window_size])
        y.append(series_values[i+window_size])
    return np.array(X), np.array(y)

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

class DLinear(nn.Module):
    def __init__(self, input_len=4, output_dim=1):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_len, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # Treino
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # Validação
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

# --------------------------------------------------
# LOOP PRINCIPAL DE EXPERIMENTOS
# --------------------------------------------------
for nc in [25, 30, 35]:
    # Treinar GMM e gerar dados sintéticos
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(dados_gmm).astype(np.float32)

    gmm = GaussianMixture(
        num_components=nc,
        covariance_type='full',
        trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1}
    )
    gmm.fit(train_scaled)

    synthetic_scaled = gmm.sample(int(len(dados_gmm) * SYNTHETIC_MULTIPLIER)).cpu().numpy()
    synthetic_df = pd.DataFrame(scaler.inverse_transform(synthetic_scaled), columns=features)
    synthetic_df['data_do_dia'] = synthetic_df['data_do_dia'].round().astype(int).apply(datetime.date.fromordinal)
    synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].round().astype(int)

    # Processar dados reais
    df_real_grouped = (
        dados[['data_do_dia', 'hora_do_dia']]
        .groupby(['data_do_dia', 'hora_do_dia'])
        .size()
        .reset_index(name='num_viagens')
    )
    df_real_grouped['datetime'] = pd.to_datetime(df_real_grouped['data_do_dia']) + pd.to_timedelta(df_real_grouped['hora_do_dia'], unit='h')
    df_real_grouped = df_real_grouped.sort_values('datetime').reset_index(drop=True)

    # Combinar dados reais e sintéticos
    df_real_plus_sint = pd.concat([
    df_real_grouped,
    synthetic_df.groupby(['data_do_dia', 'hora_do_dia']).size().reset_index(name='num_viagens')
    ]).groupby(['data_do_dia', 'hora_do_dia'], as_index=False)['num_viagens'].sum()

    df_real_plus_sint['datetime'] = (
        pd.to_datetime(df_real_plus_sint['data_do_dia']) +
        pd.to_timedelta(df_real_plus_sint['hora_do_dia'], unit='h')
    )
    df_real_plus_sint = df_real_plus_sint.sort_values('datetime').reset_index(drop=True)

    # Criar sequências
    X_real, y_real = create_sequence_dataset(df_real_grouped['num_viagens'].values.astype(float))
    X_real_sint, y_real_sint = create_sequence_dataset(df_real_plus_sint['num_viagens'].values.astype(float))

    X_real_t = torch.tensor(X_real, dtype=torch.float32)
    y_real_t = torch.tensor(y_real, dtype=torch.float32).unsqueeze(-1)
    X_real_sint_t = torch.tensor(X_real_sint, dtype=torch.float32)
    y_real_sint_t = torch.tensor(y_real_sint, dtype=torch.float32).unsqueeze(-1)

    # Loop de treinamento
    for lr in [0.001, 0.01]:
        for epochs in [50, 100, 150, 200]:
            # Dicionário para armazenar as métricas de cada dataset
            metrics_results = {}

            for data_type, (X, y) in zip(['real', 'real_synthetic'],
                                         [(X_real_t, y_real_t), (X_real_sint_t, y_real_sint_t)]):
                seed = torch.randint(0, 2**32-1, (1,)).item()
                torch.manual_seed(seed)

                split_idx = int(0.7 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                model = DLinear()
                train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=lr)

                with torch.no_grad():
                    pred = model(X_val).numpy().flatten()
                true = y_val.numpy().flatten()

                # Calcular as métricas
                metrics_values = {
                    'R²': r2_score(true, pred),
                    'SMAPE': smape(true, pred),
                    'MAE': mean_absolute_error(true, pred),
                    'MSE': mean_squared_error(true, pred),
                    'RMSE': math.sqrt(mean_squared_error(true, pred))
                }
                metrics_results[data_type] = metrics_values

            # Lista das métricas na ordem desejada
            metrics_list = ['R²', 'SMAPE', 'MAE', 'MSE', 'RMSE']
            values_real = [metrics_results['real'][metric] for metric in metrics_list]
            values_real_sint = [metrics_results['real_synthetic'][metric] for metric in metrics_list]

            # Criando o gráfico comparativo com barras lado a lado
            fig, axs = plt.subplots(3, 2, figsize=(12, 12))
            axs = axs.flatten()

            for i, metric in enumerate(metrics_list):
                ax = axs[i]
                bars = ax.bar(['Real', 'Real + Sintético'], [values_real[i], values_real_sint[i]])
                ax.set_title(metric)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', 
                            ha='center', va='bottom')

            # Caso haja algum subplot extra, removê-lo
            for j in range(len(metrics_list), len(axs)):
                fig.delaxes(axs[j])

            plt.suptitle(f"nc={nc} | lr={lr} | epochs={epochs} | seed={seed}")
            plt.tight_layout()
            plt.savefig(f"{SAVE_DIR}synthetic_{SYNTHETIC_MULTIPLIER}_nc_{nc}_lr_{lr}_epochs_{epochs}_comparacao.png")
            plt.close()
