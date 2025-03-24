import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
NUM_EXECUCOES = 10  # Número de execuções com diferentes seeds

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
dados = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek
dados = dados[dados['dia_da_semana'].between(0, 2)]
dados['data_do_dia'] = dados['tpep_pickup_datetime'].dt.date

features = [
    'data_do_dia',
    'hora_do_dia',
    'PU_longitude',
    'PU_latitude',
    'DO_longitude',
    'DO_latitude'
]
dados_gmm = dados[features].dropna().sample(frac=1.0)

# Gambiarra para poder treinar as datas do dia
dados_gmm['data_do_dia'] = pd.to_datetime(dados_gmm['data_do_dia']).apply(lambda x: x.timestamp())

# Codificação cíclica para hora_do_dia
dados_gmm['hora_sin'] = np.sin(2 * np.pi * dados_gmm['hora_do_dia'] / 24)
dados_gmm['hora_cos'] = np.cos(2 * np.pi * dados_gmm['hora_do_dia'] / 24)
dados_gmm = dados_gmm.drop(columns=['hora_do_dia'])

# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------

def create_sequence_dataset(series_values, window_size=4):
    X, y = [], []
    for i in range(len(series_values) - window_size):
        X.append(series_values[i:i+window_size])
        y.append(series_values[i+window_size])
    return np.array(X), np.array(y)

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
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

# Dataframe para armazenar resultados
df_resultados = pd.DataFrame(columns=[
    'tipo_dado', 'metrica', 'valor', 'nc', 'epochs', 'seed'
])

# --------------------------------------------------
# LOOP PRINCIPAL DE EXPERIMENTOS
# --------------------------------------------------
for nc in [25, 30, 35]:
    # 1) Treinar GMM e gerar dados sintéticos
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(dados_gmm).astype(np.float32)

    gmm = GaussianMixture(
        num_components=nc,
        covariance_type='full',
        covariance_regularization=1e-5, 
        trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1}
    )
    gmm.fit(train_scaled)

    synthetic_scaled = gmm.sample(int(len(dados_gmm) * SYNTHETIC_MULTIPLIER)).cpu().numpy()
    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=dados_gmm.columns  # Usa as colunas atualizadas
    )

    # Reconstruir 'hora_do_dia' a partir das features cíclicas
    synthetic_df['data_do_dia'] = pd.to_datetime(synthetic_df['data_do_dia'], unit='s').dt.date
    synthetic_df['hora_do_dia'] = (
        np.arctan2(synthetic_df['hora_sin'], synthetic_df['hora_cos']) * 24 / (2 * np.pi)
    ).round() % 24
    synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].astype(int)

    # 2) Processar dados reais
    df_real_grouped = (
        dados[['data_do_dia', 'hora_do_dia']]
        .groupby(['data_do_dia', 'hora_do_dia'])
        .size()
        .reset_index(name='num_viagens')
    )
    df_real_grouped['datetime'] = (
        pd.to_datetime(df_real_grouped['data_do_dia']) 
        + pd.to_timedelta(df_real_grouped['hora_do_dia'], unit='h')
    )
    df_real_grouped = df_real_grouped.sort_values('datetime').reset_index(drop=True)

    # 3) Combinar dados reais + sintéticos
    df_real_plus_sint = pd.concat([
        df_real_grouped,
        synthetic_df.groupby(['data_do_dia', 'hora_do_dia']).size().reset_index(name='num_viagens')
    ]).groupby(['data_do_dia', 'hora_do_dia'], as_index=False)['num_viagens'].sum()

    df_real_plus_sint['datetime'] = (
        pd.to_datetime(df_real_plus_sint['data_do_dia'])
        + pd.to_timedelta(df_real_plus_sint['hora_do_dia'], unit='h')
    )
    df_real_plus_sint = df_real_plus_sint.sort_values('datetime').reset_index(drop=True)

    # 4) Dados estritamente sintéticos
    df_sint = (
        synthetic_df.groupby(['data_do_dia', 'hora_do_dia'])
        .size()
        .reset_index(name='num_viagens')
    )
    df_sint['datetime'] = (
        pd.to_datetime(df_sint['data_do_dia'])
        + pd.to_timedelta(df_sint['hora_do_dia'], unit='h')
    )
    df_sint = df_sint.sort_values('datetime').reset_index(drop=True)

    # 5) Criar sequências
    X_real, y_real = create_sequence_dataset(
        df_real_grouped['num_viagens'].values.astype(float)
    )
    X_real_sint, y_real_sint = create_sequence_dataset(
        df_real_plus_sint['num_viagens'].values.astype(float)
    )
    X_sint, y_sint = create_sequence_dataset(
        df_sint['num_viagens'].values.astype(float)
    )

    # 6) Transformar em tensores
    X_real_t = torch.tensor(X_real, dtype=torch.float32)
    y_real_t = torch.tensor(y_real, dtype=torch.float32).unsqueeze(-1)

    X_real_sint_t = torch.tensor(X_real_sint, dtype=torch.float32)
    y_real_sint_t = torch.tensor(y_real_sint, dtype=torch.float32).unsqueeze(-1)

    X_sint_t = torch.tensor(X_sint, dtype=torch.float32)
    y_sint_t = torch.tensor(y_sint, dtype=torch.float32).unsqueeze(-1)

    # --------------------------------------------------
    # Loop de epochs e seeds para cada valor de nc
    # --------------------------------------------------
    for epochs in [50, 100, 150, 200]:
        execucao = 0 
        while execucao < NUM_EXECUCOES:
            seed = torch.seed() 
            torch.manual_seed(seed)
            # Vamos treinar 3 tipos de dados: real, synthetic, real+synthetic
            for data_type, (X, y) in zip(
                ['real', 'synthetic', 'real+synthetic'],
                [(X_real_t, y_real_t), (X_sint_t, y_sint_t), (X_real_sint_t, y_real_sint_t)]
            ):
                torch.manual_seed(seed)

                # treino/validação 70/30
                split_idx = int(0.7 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                model = DLinear()
                train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=0.001)

                # Previsões e métricas
                with torch.no_grad():
                    pred = model(X_val).numpy().flatten()
                true = y_val.numpy().flatten()

                metrics = {
                    'R²': r2_score(true, pred),
                    'MAE': mean_absolute_error(true, pred),
                    'RMSE': np.sqrt(mean_squared_error(true, pred))
                }

                # Salvar resultados
                for metric, value in metrics.items():
                    new_row = pd.DataFrame([{
                        'tipo_dado': data_type,
                        'metrica': metric,
                        'valor': value,
                        'nc': nc,
                        'epochs': epochs,
                        'seed': seed
                    }])
                    df_resultados = pd.concat([df_resultados, new_row], ignore_index=True)
                    
                execucao += 1

# --------------------------------------------------
# GERAÇÃO DOS BOXPLOTS (MANTIDO IGUAL)
# --------------------------------------------------

# --------------------------------------------------
# GERAÇÃO DOS BOXPLOTS
# --------------------------------------------------
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-white')

df_resultados.to_csv(f"{SAVE_DIR}resultados_metricas.csv", index=False)

metricas = ['R²', 'MAE', 'RMSE']

ncs_unicos = df_resultados['nc'].unique()

for metrica in metricas:
    df_metrica = df_resultados[df_resultados['metrica'] == metrica]

    # Loop para cada valor de nc
    for nc_atual in ncs_unicos:
        df_nc = df_metrica[df_metrica['nc'] == nc_atual]

        # Identifica os valores de epochs presentes para gerar subplots
        lista_epochs = sorted(df_nc['epochs'].unique())

        # Cálculo de colunas e linhas para subplots
        num_param = len(lista_epochs)
        cols = 3
        rows = (num_param + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols*6.5, rows*5.5))
        axs = axs.flatten()

        for i, ep in enumerate(lista_epochs):
            ax = axs[i]
            df_sub = df_nc[df_nc['epochs'] == ep]

            # Agora temos 3 tipos de dado: 'real', 'synthetic', 'real+synthetic'
            sns.boxplot(
                x='tipo_dado',
                y='valor',
                data=df_sub,
                ax=ax,
                order=['real', 'synthetic', 'real+synthetic']
            )

            ax.set_title(f'nc={nc_atual}, epochs={ep}', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metrica, fontsize=11)

            ymin, ymax = ax.get_ylim()

            # Inserir as estatísticas de média e mediana para cada tipo
            for tipo_idx, tipo in enumerate(['real', 'synthetic', 'real+synthetic']):
                dados_tipo = df_sub[df_sub['tipo_dado'] == tipo]['valor']
                if not dados_tipo.empty:
                    mediana = dados_tipo.median()
                    media = dados_tipo.mean()

                    ax.text(
                        tipo_idx, ymin,
                        f'Med: {mediana:.2f}',
                        ha='center', va='bottom',
                        color='black',
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                    )

                    ax.text(
                        tipo_idx, ymax,
                        f'Média: {media:.2f}',
                        ha='center', va='top',
                        color='darkred',
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                    )

        # Desativar eixos extras se sobrar espaço
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        plt.suptitle(
            f'Variação da métrica {metrica} - nc={nc_atual}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        plt.savefig(f"{SAVE_DIR}nc_{nc_atual}_boxplot_{metrica}.png", bbox_inches='tight')
        plt.close()
