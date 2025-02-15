import os
os.environ["CUPY_NO_PYLIBRAFT"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import pandas as pd
import numpy as np
import cupy as cp
import torch
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# --------------------------------------------
# Métricas Customizadas com CuPy
# --------------------------------------------

def silhouette_score_cupy(X, labels):
    unique_labels, labels = cp.unique(labels, return_inverse=True)
    n_clusters = len(unique_labels)
    n_samples = X.shape[0]

    if n_clusters == 1:
        return 0.0

    # Matriz de distâncias
    dist = cp.sqrt(cp.sum((X[:, cp.newaxis] - X) ** 2, axis=2))

    # Pré-calcular clusters
    cluster_masks = []
    cluster_counts = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        cluster_masks.append(mask)
        cluster_counts.append(cp.sum(mask))

    # Cálculo da distância intra-cluster (a)
    a = cp.zeros(n_samples)
    for c in range(n_clusters):
        mask = cluster_masks[c]
        count = cluster_counts[c]
        if count <= 1:
            a[mask] = 0.0
        else:
            sum_dist = cp.sum(dist[:, mask], axis=1)[mask]
            a[mask] = sum_dist / (count - 1)

    # Cálculo da distância inter-cluster (b)
    mean_dist_matrix = cp.zeros((n_samples, n_clusters))
    for c in range(n_clusters):
        mask = cluster_masks[c]
        count = cluster_counts[c]
        if count == 0:
            mean_dist_matrix[:, c] = cp.inf
        else:
            sum_dist = cp.sum(dist[:, mask], axis=1)
            mean_dist_matrix[:, c] = sum_dist / count

    # Ignorar o próprio cluster
    rows = cp.arange(n_samples)
    mean_dist_matrix[rows, labels] = cp.inf
    b = cp.min(mean_dist_matrix, axis=1)

    # Cálculo do Silhouette Score
    sil_scores = (b - a) / cp.maximum(a, b)
    sil_scores = cp.nan_to_num(sil_scores, nan=0.0)
    return float(cp.mean(sil_scores))

def calinski_harabasz_score_cupy(X, labels):
    unique_labels = cp.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = X.shape[0]

    if n_clusters <= 1:
        return 0.0

    # Centro global
    global_center = cp.mean(X, axis=0)

    # Centros e tamanhos dos clusters
    cluster_centers = []
    cluster_sizes = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        cluster = X[mask]
        cluster_centers.append(cp.mean(cluster, axis=0))
        cluster_sizes.append(cluster.shape[0])

    cluster_centers = cp.array(cluster_centers)
    cluster_sizes = cp.array(cluster_sizes)

    # Cálculo do between-cluster dispersion
    between = cp.sum(
        cluster_sizes[:, None] * (cluster_centers - global_center) ** 2
    )

    # Cálculo do within-cluster dispersion
    within = 0.0
    for i, center in enumerate(cluster_centers):
        mask = (labels == unique_labels[i])
        within += cp.sum(cp.sum((X[mask] - center) ** 2, axis=1))

    return float((between / (n_clusters - 1)) / (within / (n_samples - n_clusters)))

def davies_bouldin_score_cupy(X, labels):
    unique_labels = cp.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1:
        return 0.0

    # Centros e dispersões
    cluster_centers = []
    dispersions = []
    for lbl in unique_labels:
        mask = (labels == lbl)
        cluster = X[mask]
        center = cp.mean(cluster, axis=0)
        dispersion = cp.mean(cp.linalg.norm(cluster - center, axis=1))
        cluster_centers.append(center)
        dispersions.append(dispersion)

    cluster_centers = cp.array(cluster_centers)
    dispersions = cp.array(dispersions)

    # Matriz de distâncias entre centros
    centroids_dist = cp.sqrt(
        cp.sum((cluster_centers[:, None] - cluster_centers) ** 2, axis=2)
    )

    # Cálculo do índice
    db_scores = []
    for i in range(n_clusters):
        ratios = []
        for j in range(n_clusters):
            if i != j and centroids_dist[i, j] != 0:
                ratios.append((dispersions[i] + dispersions[j]) / centroids_dist[i, j])
        if ratios:
            db_scores.append(cp.max(cp.array(ratios)))

    return float(cp.mean(cp.array(db_scores))) if db_scores else 0.0

RANDOM_STATE = 42
COVARIANCE_TYPE = 'full'
N_JOBS = 4  
MAX_ITER = 100

def preprocessamento():
    os.makedirs('gráficos', exist_ok=True)
    print("[1] Iniciando processamento...")
    start_time = time.time()

    print("[2] Carregando e processando dados...")
    dados_taxi = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet', engine='pyarrow')
    dados_taxi['tpep_pickup_datetime'] = pd.to_datetime(dados_taxi['tpep_pickup_datetime'])

    dados_taxi['hora_do_dia'] = dados_taxi['tpep_pickup_datetime'].dt.hour
    dados_taxi['dia_da_semana'] = dados_taxi['tpep_pickup_datetime'].dt.dayofweek

    print("[3] Filtrando apenas segunda a sexta...")
    dados_taxi = dados_taxi[dados_taxi['dia_da_semana'].between(0, 4)]
    dados_taxi = dados_taxi.drop(columns=['dia_da_semana'])

    cols = [
        'hora_do_dia',
        'PU_longitude', 'PU_latitude',
        'DO_longitude', 'DO_latitude'
    ]
    dados_taxi = dados_taxi[cols].dropna()
  #  dados_taxi = dados_taxi.sample(frac=0.00001, random_state=RANDOM_STATE)

    scaler = StandardScaler()
    dados_scaled = scaler.fit_transform(dados_taxi).astype(np.float32)
    dados_scaled_cpu = torch.tensor(dados_scaled)
    
    print(f"[6] Pré-processamento concluído em {time.time() - start_time:.2f}s")
    return dados_scaled_cpu

def fit_gmm(n_clusters, data_cpu):
    with torch.cuda.device('cuda:0'):
        torch.cuda.init()
        device = torch.device("cuda")
        
        print(f"[7] Iniciando GMM com {n_clusters} clusters...")
        data_gpu = data_cpu.to(device)
        
        gmm = GaussianMixture(
            num_components=n_clusters,
            covariance_type=COVARIANCE_TYPE,
            covariance_regularization=0.0001,
            trainer_params={'max_epochs': MAX_ITER, 'accelerator': 'gpu', 'devices': 1}
        )
        gmm.fit(data_gpu)
        labels = gmm.predict(data_gpu)
        
        means = gmm.model_.means.to(device)
        diff = data_gpu - means[labels]
        inertia = torch.mean(torch.norm(diff, dim=1)).item()
        
        n_samples = data_gpu.shape[0]
        avg_log_likelihood = gmm.score(data_gpu)
        total_log_likelihood = avg_log_likelihood * n_samples
        
        n_features = data_gpu.shape[1]
        num_params = n_clusters * n_features + n_clusters * (n_features * (n_features + 1) / 2) + (n_clusters - 1)
        
        aic = 2 * num_params - 2 * total_log_likelihood
        bic = num_params * np.log(n_samples) - 2 * total_log_likelihood
        
        print(f"[8] Concluído {n_clusters} clusters | Inércia: {inertia:.2f} | AIC: {aic:.2f} | BIC: {bic:.2f}")
        return {
            'n_clusters': n_clusters,
            'inertia': inertia,
            'aic': aic,
            'bic': bic,
            'labels': labels.cpu().numpy()
        }

if __name__ == "__main__":
    dados_scaled_cpu = preprocessamento()
    clusters_range = range(10, 50)
    
    print("[9] Iniciando avaliação paralela de clusters...")
    start_fit = time.time()
    
    results = Parallel(n_jobs=N_JOBS, prefer="processes")(
        delayed(fit_gmm)(n, dados_scaled_cpu) for n in clusters_range
    )
    
    print(f"[10] Todos clusters concluídos em {time.time() - start_fit:.2f}s")
    #teste
    data_np = dados_scaled_cpu.numpy()
    n_total = len(data_np)
   # sample_size = int(0.3 * n_total)
    sample_size = int(1.0 * n_total)
    
    np.random.seed(RANDOM_STATE)
    sample_indices = np.random.choice(n_total, size=sample_size, replace=False)
    data_sample = data_np[sample_indices]
    
    # Conversão para CuPy
    data_sample_cupy = cp.asarray(data_sample)
    
for res in results:
        labels = res['labels']
        labels_sample = labels[sample_indices]
        print(f"Calculando métricas para {res['n_clusters']} clusters...")

        # Calcular métricas em batches
        batch_size = 10000  # Defina um tamanho de batch adequado
        n_batches = (len(data_sample) + batch_size - 1) // batch_size

        silhouette_scores = []
        ch_scores = []
        db_scores = []

        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(data_sample))
            batch_data = data_sample[start:end]
            batch_labels = labels_sample[start:end]

            silhouette_scores.append(silhouette_score_cupy(cp.asarray(batch_data), cp.asarray(batch_labels)))
            ch_scores.append(calinski_harabasz_score_cupy(cp.asarray(batch_data), cp.asarray(batch_labels)))
            db_scores.append(davies_bouldin_score_cupy(cp.asarray(batch_data), cp.asarray(batch_labels)))

            print(f"Batch {i+1}/{n_batches} processado.")  # Print batch progress

        # Add the calculated scores to the dictionary 'res'
        res['silhouette'] = np.mean(silhouette_scores)
        res['calinski_harabasz'] = np.mean(ch_scores)
        res['davies_bouldin'] = np.mean(db_scores)
        print(f"Métricas calculadas para {res['n_clusters']} clusters.")

results = sorted(results, key=lambda x: x['n_clusters'])
# Salvar métricas em arquivo
with open('metricas_clusters.txt', 'w') as f:
    f.write("n_clusters, inertia, aic, bic, silhouette, calinski_harabasz, davies_bouldin\n")
    for i, res in enumerate(results):
        f.write(f"{res['n_clusters']}, {res['inertia']}, {res['aic']}, {res['bic']}, {res['silhouette']}, {res['calinski_harabasz']}, {res['davies_bouldin']}\n")

# Plotagem
n_clusters_list = [r['n_clusters'] for r in results]
inertia_list = [r['inertia'] for r in results]
aic_list = [r['aic'] for r in results]
bic_list = [r['bic'] for r in results]
silhouette_list = [r['silhouette'] for r in results]
ch_list = [r['calinski_harabasz'] for r in results]
db_list = [r['davies_bouldin'] for r in results]

plt.figure(figsize=(18, 12))

# Inércia vs Número de Clusters
plt.subplot(2, 3, 1)
plt.plot(n_clusters_list, inertia_list, marker='o', color='blue')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia Média')
plt.title('Inércia vs Número de Clusters')
plt.grid(True)

# AIC vs Número de Clusters
plt.subplot(2, 3, 2)
plt.plot(n_clusters_list, aic_list, marker='o', color='green')
plt.xlabel('Número de Clusters')
plt.ylabel('AIC')
plt.title('AIC vs Número de Clusters')
plt.grid(True)

# BIC vs Número de Clusters
plt.subplot(2, 3, 3)
plt.plot(n_clusters_list, bic_list, marker='o', color='red')
plt.xlabel('Número de Clusters')
plt.ylabel('BIC')
plt.title('BIC vs Número de Clusters')
plt.grid(True)

# Silhouette vs Número de Clusters
plt.subplot(2, 3, 4)
plt.plot(n_clusters_list, silhouette_list, marker='o', color='purple')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette')
plt.title('Silhouette vs Número de Clusters')
plt.grid(True)

# Calinski-Harabasz vs Número de Clusters
plt.subplot(2, 3, 5)
plt.plot(n_clusters_list, ch_list, marker='s', color='orange')
plt.xlabel('Número de Clusters')
plt.ylabel('Calinski-Harabasz')
plt.title('Calinski-Harabasz vs Número de Clusters')
plt.grid(True)

# Davies-Bouldin vs Número de Clusters
plt.subplot(2, 3, 6)
plt.plot(n_clusters_list, db_list, marker='^', color='brown')
plt.xlabel('Número de Clusters')
plt.ylabel('Davies-Bouldin')
plt.title('Davies-Bouldin vs Número de Clusters')
plt.grid(True)

plt.tight_layout()
plt.savefig('gráficos/avaliacao_clusters_gmm.png', dpi=300)
plt.show()

print("[11] Gráficos gerados e salvos em 'gráficos/avaliacao_clusters_gmm.png'")
print("[12] Métricas salvas em 'metricas_clusters.txt'")
