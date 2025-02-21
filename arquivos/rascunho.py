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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
import umap.umap_ as umap
from matplotlib.patches import Ellipse

def cdist_gpu_custom(X, Y):
    X_norm_sq = cp.sum(X ** 2, axis=1)[:, None]
    Y_norm_sq = cp.sum(Y ** 2, axis=1)[None, :]
    dist_sq = X_norm_sq + Y_norm_sq - 2 * cp.dot(X, Y.T)
    dist_sq = cp.maximum(dist_sq, 0)
    return cp.sqrt(dist_sq)

RANDOM_STATE = 42
COVARIANCE_TYPE = 'full'
MAX_ITER = 100

def preprocessamento(start_day, end_day, day_type):
    os.makedirs('gráficos', exist_ok=True)
    print(f"[1] Iniciando processamento para {day_type}...")
    start_time = time.time()

    print("[2] Carregando e processando dados...")
    dados_taxi = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet', engine='pyarrow')
    dados_taxi['tpep_pickup_datetime'] = pd.to_datetime(dados_taxi['tpep_pickup_datetime'])
    dados_taxi['hora_do_dia'] = dados_taxi['tpep_pickup_datetime'].dt.hour
    dados_taxi['dia_da_semana'] = dados_taxi['tpep_pickup_datetime'].dt.dayofweek

    print(f"[3] Filtrando {day_type}...")
    # Para dias úteis: de 0 a 4; para fim de semana: dias 5 e 6 (a condição abaixo usa 5 a 8, mas apenas 5 e 6 serão válidos)
    dados_taxi = dados_taxi[dados_taxi['dia_da_semana'].between(start_day, end_day)]
   # dados_taxi = dados_taxi.sample(frac=0.5, random_state=RANDOM_STATE)
    
    cols_modelo = ['hora_do_dia', 'PU_longitude', 'PU_latitude', 'DO_longitude', 'DO_latitude']
    cols_ids = ['PULocationID', 'DOLocationID']
    dados_taxi = dados_taxi[cols_modelo + cols_ids].dropna().reset_index(drop=True)

    scaler = StandardScaler()
    dados_scaled = scaler.fit_transform(dados_taxi[cols_modelo]).astype(np.float32)
    dados_scaled_cpu = torch.tensor(dados_scaled)
    
    print(f"[4] Pré-processamento concluído em {time.time() - start_time:.2f}s")
    return dados_scaled_cpu, dados_taxi

def fit_gmm(n_clusters, data_cpu):
    with torch.cuda.device('cuda:0'):
        torch.cuda.init()
        device = torch.device("cuda")
        print(f"[5] Iniciando GMM com {n_clusters} clusters...")
        data_gpu = data_cpu.to(device)
        
        gmm = GaussianMixture(
            num_components=n_clusters,
            covariance_type=COVARIANCE_TYPE,
            covariance_regularization=0.0001,
            trainer_params={'max_epochs': MAX_ITER, 'accelerator': 'gpu', 'devices': 1}
        )
        gmm.fit(data_gpu)
        labels = gmm.predict(data_gpu)
        labels = labels.cpu().numpy()
        print(f"[6] GMM concluído com {n_clusters} clusters.")
        return labels, gmm

if __name__ == "__main__":
    # Processa primeiro os dias úteis e depois o final de semana
    for config in [(0, 4, 'dia_util'), (5, 8, 'fds')]:
        start_day, end_day, suffix = config
        
        # Define o número de clusters conforme o período
        if suffix == 'dia_util':
            best_n = 11
        elif suffix == 'fds':
            best_n = 15

        # Pré-processamento dos dados conforme o tipo de dia
        dados_scaled_cpu, dados_original = preprocessamento(start_day, end_day, suffix)
        
        labels, gmm = fit_gmm(best_n, dados_scaled_cpu)
        
        # Seleciona uma amostra dos dados para reduzir o tempo de cálculo das métricas
        sample_fraction = 0.001
        sample_size = int(len(dados_scaled_cpu) * sample_fraction)
        sample_indices = np.random.choice(len(dados_scaled_cpu), size=sample_size, replace=False)
        data_sample = dados_scaled_cpu.numpy()[sample_indices]
        sample_labels = labels[sample_indices]

        # MDS
        print("[7] Aplicando MDS...")
        start_mds = time.time()
        data_mds = MDS(n_components=2, random_state=RANDOM_STATE, n_jobs=-1).fit_transform(data_sample)
        
        plt.figure(figsize=(16,8))
        plt.scatter(data_mds[:,0], data_mds[:,1], c=sample_labels, cmap='tab20', alpha=0.6, s=20)
        plt.title(f'MDS - {best_n} Clusters ({suffix})')
        plt.savefig(f'gráficos/mds_{best_n}_{suffix}.png', dpi=300)
        plt.close()
        print(f"MDS concluído em {time.time() - start_mds:.2f}s")

        # t-SNE
        print("[8] Aplicando t-SNE...")
        start_tsne = time.time()
        data_tsne = TSNE(n_components=2, random_state=RANDOM_STATE, n_jobs=-1).fit_transform(data_sample)
        
        plt.figure(figsize=(16,8))
        plt.scatter(data_tsne[:,0], data_tsne[:,1], c=sample_labels, cmap='tab20', alpha=0.6, s=20)
        plt.title(f't-SNE - {best_n} Clusters ({suffix})')
        plt.savefig(f'gráficos/tsne_{best_n}_{suffix}.png', dpi=300)
        plt.close()
        print(f"t-SNE concluído em {time.time() - start_tsne:.2f}s")

        # UMAP
        print("[9] Aplicando UMAP...")
        start_umap = time.time()
        data_umap = umap.UMAP(random_state=RANDOM_STATE, n_jobs=-1).fit_transform(data_sample)
        
        plt.figure(figsize=(16,8))
        plt.scatter(data_umap[:,0], data_umap[:,1], c=sample_labels, cmap='tab20', alpha=0.6, s=20)
        plt.title(f'UMAP - {best_n} Clusters ({suffix})')
        plt.savefig(f'gráficos/umap_{best_n}_{suffix}.png', dpi=300)
        plt.close()
        print(f"UMAP concluído em {time.time() - start_umap:.2f}s")

        # PCA + t-SNE
        print("[10] Aplicando PCA + t-SNE...")
        start_pca_tsne = time.time()
        data_pca = PCA(n_components=3).fit_transform(data_sample)
        data_pca_tsne = TSNE(n_components=2, random_state=RANDOM_STATE, n_jobs=-1).fit_transform(data_pca)
        
        plt.figure(figsize=(16,8))
        plt.scatter(data_pca_tsne[:,0], data_pca_tsne[:,1], c=sample_labels, cmap='tab20', alpha=0.6, s=20)
        plt.title(f'PCA + t-SNE - {best_n} Clusters ({suffix})')
        plt.savefig(f'gráficos/pca_tsne_{best_n}_{suffix}.png', dpi=300)
        plt.close()
        print(f"PCA+t-SNE concluído em {time.time() - start_pca_tsne:.2f}s")

        # PCA + UMAP
        print("[11] Aplicando PCA + UMAP...")
        start_pca_umap = time.time()
        data_pca = PCA(n_components=3).fit_transform(data_sample)
        data_pca_umap = umap.UMAP(random_state=RANDOM_STATE, n_jobs=-1).fit_transform(data_pca)
        
        plt.figure(figsize=(16,8))
        plt.scatter(data_pca_umap[:,0], data_pca_umap[:,1], c=sample_labels, cmap='tab20', alpha=0.6, s=20)
        plt.title(f'PCA + UMAP - {best_n} Clusters ({suffix})')
        plt.savefig(f'gráficos/pca_umap_{best_n}_{suffix}.png', dpi=300)
        plt.close()
        print(f"PCA+UMAP concluído em {time.time() - start_pca_umap:.2f}s")

    print("[12] Todos os gráficos foram gerados na pasta 'gráficos'.")
