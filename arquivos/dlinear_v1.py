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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
dados = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek

# Filtrar apenas segunda(0) a quarta(2)
dados = dados[dados['dia_da_semana'].between(0, 2)]

dados['data_do_dia'] = dados['tpep_pickup_datetime'].dt.date


features = ['data_do_dia', 'hora_do_dia', 'PU_longitude', 'PU_latitude', 'DO_longitude', 'DO_latitude']
dados_gmm = dados[features].dropna().sample(frac=1.0, random_state=42)
dados_gmm['data_do_dia'] = pd.to_datetime(dados_gmm['data_do_dia']).apply(lambda x: x.toordinal()) #transformação linear na data
print("Dados carregados e filtrados. Total de registros (para GMM):", len(dados_gmm))

# --------------------------------------------------
# 2) TREINAR GMM (APENAS COM DADOS REAIS) E GERAR AMOSTRAS SINTÉTICAS
# --------------------------------------------------
nc = 30                   # número de componentes do GMM
SYNTHETIC_MULTIPLIER = 0.5  # quantas vezes o tamanho real para dados sintéticos 
# Escalando antes do GMM
scaler = StandardScaler()
train_scaled = scaler.fit_transform(dados_gmm).astype(np.float32)

print(f"Treinando GMM com n_clusters = {nc}...")
gmm = GaussianMixture(
    num_components=nc,
    covariance_type='full',
    trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1}
)
gmm.fit(train_scaled)
print("GMM treinado.")

# Gera amostras sintéticas
num_sint_samples = len(dados_gmm) * SYNTHETIC_MULTIPLIER
synthetic_scaled = gmm.sample(num_sint_samples).cpu().numpy()  # array escalado
synthetic_data = scaler.inverse_transform(synthetic_scaled)    # desfaz o scaling
synthetic_df = pd.DataFrame(synthetic_data, columns=features)
synthetic_df['data_do_dia'] = synthetic_df['data_do_dia'].round().astype(int).apply(datetime.date.fromordinal)

synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].round().astype(int)
#print(synthetic_df.head())

# --------------------------------------------------
# 3) AGRUPAR DADOS REAIS E SINTÉTICOS (AGORA POR DIA + HORA)
# --------------------------------------------------

df_real = dados[['data_do_dia', 'hora_do_dia']].dropna()

df_real_grouped = (
    df_real
    .groupby(['data_do_dia', 'hora_do_dia'])
    .size()
    .reset_index(name='num_viagens')
)

# Para o sintético, fazemos algo análogo
df_sint_grouped = (
    synthetic_df
    .groupby(['data_do_dia', 'hora_do_dia'])
    .size()
    .reset_index(name='num_viagens')
)

# Combinar real + sintético
df_real_plus_sint = pd.concat([df_real_grouped, df_sint_grouped], ignore_index=True)
df_real_plus_sint = (
    df_real_plus_sint
    .groupby(['data_do_dia', 'hora_do_dia'])
    .agg({'num_viagens':'sum'})
    .reset_index()
)

# Real
df_real_grouped['datetime'] = (
    pd.to_datetime(df_real_grouped['data_do_dia']) +
    pd.to_timedelta(df_real_grouped['hora_do_dia'], unit='h')
)
df_real_grouped = df_real_grouped.sort_values('datetime').reset_index(drop=True)

# Real + sintético
df_real_plus_sint['datetime'] = (
    pd.to_datetime(df_real_plus_sint['data_do_dia']) +
    pd.to_timedelta(df_real_plus_sint['hora_do_dia'], unit='h')
)
df_real_plus_sint = df_real_plus_sint.sort_values('datetime').reset_index(drop=True)

# --------------------------------------------------
# 4) CRIAR O CONJUNTO DE TREINO PARA O DLINEAR
# --------------------------------------------------
def create_sequence_dataset(series_values, window_size=4):
    """
    Cria janelas deslizantes de tamanho 'window_size' para predição univariada.
    X: [n_exemplos, window_size]
    y: [n_exemplos, 1]
    """
    X, y = [], []
    for i in range(len(series_values) - window_size):
        X.append(series_values[i : i + window_size])
        y.append(series_values[i + window_size])
    return np.array(X), np.array(y)

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100 

# 4.1) Dados Reais
real_series = df_real_grouped['num_viagens'].values.astype(float)
X_real, y_real = create_sequence_dataset(real_series, window_size=4)

# 4.2) Dados Reais + Sintéticos
real_sint_series = df_real_plus_sint['num_viagens'].values.astype(float)
X_real_sint, y_real_sint = create_sequence_dataset(real_sint_series, window_size=4)

# Transforma em tensores
X_real_t = torch.tensor(X_real, dtype=torch.float32)
y_real_t = torch.tensor(y_real, dtype=torch.float32).unsqueeze(-1)

X_real_sint_t = torch.tensor(X_real_sint, dtype=torch.float32)
y_real_sint_t = torch.tensor(y_real_sint, dtype=torch.float32).unsqueeze(-1)

# Splits de treino/val 
split_idx_real = int(0.7 * len(X_real_t))
Xr_train = X_real_t[:split_idx_real]
Xr_val   = X_real_t[split_idx_real:]
yr_train = y_real_t[:split_idx_real]
yr_val   = y_real_t[split_idx_real:]

split_idx_sint = int(0.7 * len(X_real_sint_t))
Xrs_train = X_real_sint_t[:split_idx_sint]
Xrs_val   = X_real_sint_t[split_idx_sint:]
yrs_train = y_real_sint_t[:split_idx_sint]
yrs_val   = y_real_sint_t[split_idx_sint:]

print("Tamanho treino REAL:", Xr_train.shape, yr_train.shape)
print("Tamanho treino REAL+SINT:", Xrs_train.shape, yrs_train.shape)

# --------------------------------------------------
# 5) DEFINIR O MODELO DLINEAR E FUNÇÃO DE TREINO
# --------------------------------------------------
class DLinear(nn.Module):
    """
    Implementação simples de um modelo DLinear (univariado),
    que projeta a janela de entrada diretamente em 1 output.
    """
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
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Val
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = criterion(y_val_pred, y_val)

        if (epoch+1) % 10 == 0:
            print(f"Epoch[{epoch+1}/{epochs}] - Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

# --------------------------------------------------
# 6) TREINAR E COMPARAR OS DOIS MODELOS
# --------------------------------------------------
# 6.1) Modelo com dados REAIS
torch.manual_seed(42)
model_real = DLinear(input_len=4, output_dim=1)
train_model(model_real, Xr_train, yr_train, Xr_val, yr_val, epochs=50, lr=0.001)

model_real.eval()
pred_real = model_real(Xr_val).detach().numpy().flatten()
true_real = yr_val.numpy().flatten()

#métrica dados reais
r2_real = r2_score(true_real, pred_real)
mae_real  = mean_absolute_error(true_real, pred_real)
mse_real  = mean_squared_error(true_real, pred_real)
rmse_real = math.sqrt(mse_real)
smape_real = smape(true_real, pred_real)
print(f"SMAPE (somente real): {smape_real:.4f}")
print(f"\nR2 (somente real)  : {r2_real:.4f}")
print(f"MAE (somente real)  : {mae_real:.4f}")
print(f"MSE (somente real)  : {mse_real:.4f}")
print(f"RMSE (somente real) : {rmse_real:.4f}")

# 6.2) Modelo com dados REAIS + SINTÉTICOS
torch.manual_seed(42)
model_real_sint = DLinear(input_len=4, output_dim=1)
train_model(model_real_sint, Xrs_train, yrs_train, Xrs_val, yrs_val, epochs=600, lr=0.001)

model_real_sint.eval()
pred_real_sint = model_real_sint(Xrs_val).detach().numpy().flatten()
true_real_sint = yrs_val.numpy().flatten()

#métricas dataset real + sintético
r2_real_sint = r2_score(true_real_sint, pred_real_sint)
mae_real_sint  = mean_absolute_error(true_real_sint, pred_real_sint)
mse_real_sint  = mean_squared_error(true_real_sint, pred_real_sint)
rmse_real_sint = math.sqrt(mse_real_sint)
smape_real_sint = smape(true_real_sint, pred_real_sint)
print(f"SMAPE (real + sint): {smape_real_sint:.4f}")
print(f"R2 (real + sint)   : {r2_real_sint:.4f}")
print(f"MAE (real + sint)   : {mae_real_sint:.4f}")
print(f"MSE (real + sint)   : {mse_real_sint:.4f}")
print(f"RMSE (real + sint)  : {rmse_real_sint:.4f}")

#plota gráfico
metrics = ["R²", "SMAPE", "MAE", "MSE", "RMSE"]
values_real = [r2_real, smape_real, mae_real, mse_real, rmse_real]
values_real_sint = [r2_real_sint, smape_real_sint, mae_real_sint, mse_real_sint, rmse_real_sint]

# Criando uma figura com uma grade de subplots (3 linhas x 2 colunas)
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()  # Facilita a iteração

# Plotando cada métrica em um subplot
for i, metric in enumerate(metrics):
    ax = axs[i]
    # Cada subplot terá duas barras: uma para "Somente Real" e outra para "Real + Sint"
    bars = ax.bar(['Somente Real', 'Real + Sint'], [values_real[i], values_real_sint[i]])
    ax.set_title(metric)
    ax.set_ylabel("Valor")
    # Exibe os valores das métricas acima de cada barra
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')

for j in range(len(metrics), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.savefig("/home/caioloss/gráficos/linear/comparacao_metricas_v1.png")  
