import os
os.environ["CUPY_NO_PYLIBRAFT"] = "1"
import time
import pandas as pd
import numpy as np
import cupy as cp
from cupyx.scipy.spatial.distance import pdist as cp_pdist
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from pytorch_lightning import Trainer


# --------------------------------------------
# Configurações Globais
# --------------------------------------------
BEST_N = 33
COVARIANCE_TYPE = 'full'
N_JOBS = -1           # Usa todos os núcleos disponíveis
MAX_ITER = 100        # Máximo de iterações do GMM
RANDOM_STATE = 42

# --------------------------------------------
# Pré-processamento
# --------------------------------------------
os.makedirs('gráficos', exist_ok=True)
print("Iniciando processamento...")
start_time = time.time()

print("Carregando e processando dados...")
# Leitura do arquivo parquet utilizando o engine 'pyarrow' para melhor desempenho
dados_taxi = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet', engine='pyarrow')
dados_taxi['tpep_pickup_datetime'] = pd.to_datetime(dados_taxi['tpep_pickup_datetime'])

# Criação de colunas adicionais
dados_taxi['hora_do_dia'] = dados_taxi['tpep_pickup_datetime'].dt.hour
dados_taxi['dia_da_semana'] = dados_taxi['tpep_pickup_datetime'].dt.dayofweek


# Seleciona colunas e remove valores nulos
cols = [
    'hora_do_dia', 'dia_da_semana',
    'PU_longitude', 'PU_latitude',
    'DO_longitude', 'DO_latitude',
    'PULocationID', 'DOLocationID'
]
dados_taxi = dados_taxi[cols].dropna()

# Filtra apenas PULocationIDs "válidos"
valid_pu_ids = {
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
    103, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142,
    143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194,
    202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
    244, 246, 249, 261, 262, 263
}
dados_taxi = dados_taxi[dados_taxi['PULocationID'].isin(valid_pu_ids)]

# Seleciona aleatoriamente 50% das linhas
dados_taxi = dados_taxi.sample(frac=0.5, random_state=RANDOM_STATE)

# Escala as variáveis contínuas
scaler = StandardScaler()
dados_scaled = scaler.fit_transform(dados_taxi[[
    'hora_do_dia', 'dia_da_semana',
    'PU_longitude', 'PU_latitude',
    'DO_longitude', 'DO_latitude'
]]).astype(np.float32)

print(f"[1] Pré-processamento concluído em {time.time() - start_time:.2f}s")

# --------------------------------------------
# Ajuste do GMM usando best_n = 33
# --------------------------------------------
start_gmm = time.time()
print(f"\nIniciando fit do GMM com n_components={BEST_N}...")

gmm = GaussianMixture(
    num_components=BEST_N,
    covariance_type=COVARIANCE_TYPE,
    covariance_regularization= 0.0001
)
gmm.fit(dados_scaled)
labels = gmm.predict(dados_scaled)

# Converte os rótulos para array NumPy, se forem um tensor
if hasattr(labels, 'cpu'):
    labels = labels.cpu().numpy()

print(f"Fit concluído em {time.time() - start_gmm:.2f}s")
# --------------------------------------------
# Cálculo da distância média dentro de cada cluster utilizando GPU (Otimizado)
# --------------------------------------------

print("\nIniciando cálculo de distâncias médias com GPU (processamento em lotes)...")

def compute_mean_distance_gpu(cluster_data, batch_size=1000):
    """
    Calcula a distância média entre pontos em um cluster processando em lotes para evitar estouro de memória.
    """
    N = cluster_data.shape[0]
    if N < 2:
        return 0.0

    # Pré-calcula as normas quadradas dos pontos uma vez
    norms = cp.sum(cluster_data ** 2, axis=1)

    total_sum = 0.0
    total_pairs = 0

    # Processa cada lote de pontos
    for i_start in range(0, N, batch_size):
        i_end = i_start + batch_size
        current_batch = cluster_data[i_start:i_end]
        current_norms = norms[i_start:i_end]
        current_size = current_batch.shape[0]

        # Calcula distâncias dentro do lote atual
        if current_size > 1:
            dot_product = cp.dot(current_batch, current_batch.T)
            dists_sq = current_norms[:, None] + current_norms[None, :] - 2 * dot_product
            dists_sq = cp.maximum(dists_sq, 0)
            iu = cp.triu_indices(current_size, k=1)
            within_dists = cp.sqrt(dists_sq[iu])
            total_sum += cp.sum(within_dists).get()  # Move para CPU
            total_pairs += within_dists.size

        # Processa pontos restantes após o lote atual
        rest_start = i_end
        if rest_start >= N:
            continue

        # Processa o restante em lotes menores
        rest_batch_size = batch_size
        for r_start in range(rest_start, N, rest_batch_size):
            r_end = r_start + rest_batch_size
            rest_batch = cluster_data[r_start:r_end]
            rest_norms = norms[r_start:r_end]

            # Calcula distâncias entre o lote atual e o lote restante
            dot_product = cp.dot(current_batch, rest_batch.T)
            dists_sq = current_norms[:, None] + rest_norms[None, :] - 2 * dot_product
            dists_sq = cp.maximum(dists_sq, 0)
            dists = cp.sqrt(dists_sq)
            sum_dists = cp.sum(dists)
            total_sum += sum_dists.get()  # Acumula no CPU
            total_pairs += current_size * rest_batch.shape[0]

    return total_sum / total_pairs if total_pairs > 0 else 0.0

start_distance = time.time()

# Converte dados para GPU
dados_scaled_gpu = cp.asarray(dados_scaled)
labels_gpu = cp.asarray(labels)

mean_distances = []
for cluster_id in range(BEST_N):
    # Obtém índices diretamente para economizar memória
    cluster_indices = cp.where(labels_gpu == cluster_id)[0]
    cluster_data = dados_scaled_gpu[cluster_indices]
    
    if cluster_data.shape[0] < 2:
        mean_distances.append(0.0)
        continue
        
    # Calcula com tamanho de lote ajustado para 500 para maior segurança
    mean_dist = compute_mean_distance_gpu(cluster_data, batch_size=500)
    mean_distances.append(mean_dist)

# Salva resultados
distancias_df = pd.DataFrame({
    'cluster': range(BEST_N),
    'mean_distance': mean_distances
})
distancias_df.to_csv('gráficos/distancias_medias_gmm_n33.txt', sep='\t', index=False)

print(f"Cálculo de distâncias concluído em {time.time() - start_distance:.2f}s")
print("Distâncias médias salvas em 'gráficos/distancias_medias_gmm_n33.txt'")