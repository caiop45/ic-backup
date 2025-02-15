import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from joblib import Parallel, delayed

# Carrega os arquivos e limpa os dados
tempo_inicio = time.time()

dados_taxi = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados_taxi['tpep_pickup_datetime'] = pd.to_datetime(dados_taxi['tpep_pickup_datetime'])

dados_taxi['hora_do_dia'] = dados_taxi['tpep_pickup_datetime'].dt.hour
dados_taxi['dia_da_semana'] = dados_taxi['tpep_pickup_datetime'].dt.dayofweek  # 0=Segunda, 2=Quarta

cols = ['hora_do_dia', 'dia_da_semana',
        'PU_longitude', 'PU_latitude',
        'DO_longitude', 'DO_latitude',
        'PULocationID', 'DOLocationID']

dados_taxi = dados_taxi[cols].dropna()

# Normalização dos dados
scaler = StandardScaler()
dados_scaled = scaler.fit_transform(dados_taxi[['hora_do_dia', 'dia_da_semana',
                                              'PU_longitude', 'PU_latitude',
                                              'DO_longitude', 'DO_latitude']])

print(f"[1] Pré-processamento concluído em {time.time() - tempo_inicio:.2f}s")

# Fit do GMM 
tempo_inicio = time.time()
best_n = 27
gmm = GaussianMixture(n_components=best_n, covariance_type='full', random_state=42)
gmm.fit(dados_scaled)
labels = gmm.predict(dados_scaled)
dados_taxi['cluster'] = labels
print(f"[2] GMM treinado em {time.time() - tempo_inicio:.2f}s")

# =================================================================
# SEÇÃO MODIFICADA: Paralelismo no cálculo da distância euclidiana média
# =================================================================
os.makedirs('gráficos', exist_ok=True)

# Filtrar viagens às 18h e quartas-feiras
dados_filtrados = dados_taxi[(dados_taxi['hora_do_dia'] == 18) &
                             (dados_taxi['dia_da_semana'] == 2)]

# Função para calcular a distância euclidiana média para um cluster
def calculate_mean_distance(cluster_id, data, scaler):
    cluster_data = data[data['cluster'] == cluster_id]
    cluster_scaled = scaler.transform(cluster_data[['hora_do_dia', 'dia_da_semana',
                                                    'PU_longitude', 'PU_latitude',
                                                    'DO_longitude', 'DO_latitude']])
    distances = euclidean_distances(cluster_scaled)
    mean_distance = np.mean(distances[~np.eye(distances.shape[0], dtype=bool)])
    return {'cluster': cluster_id, 'mean_euclidean_distance': mean_distance}

# Paralelizar o cálculo da distância euclidiana média
n_jobs = -1  # Usar todos os núcleos disponíveis
mean_euclidean_distances = Parallel(n_jobs=n_jobs)(
    delayed(calculate_mean_distance)(cluster_id, dados_filtrados, scaler)
    for cluster_id in dados_filtrados['cluster'].unique()
)

# Criar DataFrame com as distâncias médias
distancias_medias_df = pd.DataFrame(mean_euclidean_distances)

# Adicionar a informação de distância média ao DataFrame original
dados_filtrados = pd.merge(dados_filtrados, distancias_medias_df, on='cluster', how='left')

# Seleciona os 10 clusters com mais viagens no filtro
top_clusters = dados_filtrados['cluster'].value_counts().head(10).index.tolist()
dados_top = dados_filtrados[dados_filtrados['cluster'].isin(top_clusters)]

# Ordenar os clusters pela distância euclidiana média
top_clusters_sorted = distancias_medias_df.sort_values('mean_euclidean_distance')['cluster'].head(10).tolist()
dados_top = dados_filtrados[dados_filtrados['cluster'].isin(top_clusters_sorted)]

# Exibir as distâncias euclidianas médias para os 10 clusters com mais viagens
print("\nDistâncias Euclidianas Médias para os 10 clusters com mais viagens:")
print(distancias_medias_df[distancias_medias_df['cluster'].isin(top_clusters)].sort_values('mean_euclidean_distance'))
exit()
# Configurações do gráfico
plt.figure(figsize=(16, 10))
palette = sns.color_palette("tab20", n_colors=len(top_clusters_sorted))

sns.scatterplot(
    data=dados_top,
    x='PULocationID',
    y='DOLocationID',
    hue='cluster',
    palette=palette,
    alpha=0.7,
    s=50,
    edgecolor='none'
)

plt.title("Top 10 Clusters Ordenados por Distância Euclidiana Média: Quartas-Feiras às 18h", fontsize=14)
plt.xlabel("Zona de Embarque (PULocationID)", fontsize=12)
plt.ylabel("Zona de Desembarque (DOLocationID)", fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=8)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('gráficos/top10_clusters_quarta_18h_ordenados_distancia_paralelo.png', dpi=300, bbox_inches='tight')
plt.close()

print("[3] Visualização salva em 'gráficos/top10_clusters_quarta_18h_ordenados_distancia_paralelo.png'")
