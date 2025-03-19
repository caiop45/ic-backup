import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np

# =============================
# 1. Leitura e Pré-Processamento
# =============================
tempo_inicio = time.time()

# Carrega os dados (certifique-se de incluir PULocationID e DOLocationID)
dados_taxi = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados_taxi['tpep_pickup_datetime'] = pd.to_datetime(dados_taxi['tpep_pickup_datetime'])

# Cria features temporais
dados_taxi['hora_do_dia'] = dados_taxi['tpep_pickup_datetime'].dt.hour
dados_taxi['dia_da_semana'] = dados_taxi['tpep_pickup_datetime'].dt.dayofweek
dados_taxi = dados_taxi[dados_taxi['dia_da_semana'].between(1, 3)]

# Seleciona colunas relevantes (inclua PULocationID e DOLocationID se disponíveis)
cols = ['hora_do_dia', 
        'PU_longitude', 'PU_latitude', 
        'DO_longitude', 'DO_latitude',
        'PULocationID', 'DOLocationID']  # Adicione estas colunas!

dados_taxi = dados_taxi[cols].dropna()

# =============================
# 2. Normalização dos Dados
# =============================
scaler = StandardScaler()
dados_scaled = scaler.fit_transform(dados_taxi[['hora_do_dia', 'dia_da_semana',
                                              'PU_longitude', 'PU_latitude',
                                              'DO_longitude', 'DO_latitude']])

print(f"[1] Pré-processamento concluído em {time.time() - tempo_inicio:.2f}s")

# =============================
# 3. Seleção do Número de Clusters (BIC/AIC)
# =============================
n_components_range = range(10, 30)

tempo_inicio = time.time()
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, covariance_type='diag', random_state=42)
    gmm.fit(dados_scaled)
    bic.append(gmm.bic(dados_scaled))
    aic.append(gmm.aic(dados_scaled))
    print(bic)
    print(aic)

# Plota BIC/AIC
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic, label='BIC', marker='o')
plt.plot(n_components_range, aic, label='AIC', marker='s')
plt.xlabel('Número de Clusters')
plt.ylabel('Valor do Critério')
plt.title('Seleção de Número de Clusters via BIC/AIC')
plt.legend()
plt.savefig('gráficos/bic_aic.png', dpi=120)
print(f"[2] Seleção de clusters concluída em {time.time() - tempo_inicio:.2f}s")
#Escolhe o melhor n (exemplo: mínimo do BIC)
best_n = n_components_range[np.argmin(bic)]
print(best_n)