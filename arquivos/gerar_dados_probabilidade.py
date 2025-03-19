import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# =============================================================================
# Dispositivo e otimizações
# =============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# =============================================================================
# Modelo DLinear
# =============================================================================
class DLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# =============================================================================
# Funções de avaliação
# =============================================================================
def evaluate_model(model, loader, hour_mean, hour_std):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs).squeeze().cpu().numpy()
            y_pred.extend(outputs)
            y_true.extend(targets.numpy())
    y_true = np.array(y_true) * hour_std + hour_mean
    y_pred = np.array(y_pred) * hour_std + hour_mean
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred)**2)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'mse': mse, 'r2': r2}

# =============================================================================
# Função de treino (com validação e early stopping)
# =============================================================================
def train_with_validation(
    model,
    train_loader,
    val_loader,
    hour_mean,
    hour_std,
    epochs=10,
    lr=0.001,
    patience=3,
    is_finetune=False
):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_loss = float('inf')
    best_state_dict = None
    no_improve_counter = 0

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Avalia no conjunto de validação
        val_metrics = evaluate_model(model, val_loader, hour_mean, hour_std)
        val_loss = val_metrics['mse']  # Early stopping baseado em MSE

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                break

    # Restaura o melhor estado
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

# =============================================================================
# Métricas entre distribuições (utilizando CuPy)
# =============================================================================
def calcular_metricas(real, sintetico, num_bins=50):
    min_val = cp.minimum(cp.min(real), cp.min(sintetico))
    max_val = cp.maximum(cp.max(real), cp.max(sintetico))
    bins = cp.linspace(min_val, max_val, num_bins + 1)
    hist_real, _ = cp.histogram(real, bins=bins, density=True)
    hist_sint, _ = cp.histogram(sintetico, bins=bins, density=True)
    hist_real = cp.where(hist_real == 0, 1e-8, hist_real)
    hist_sint = cp.where(hist_sint == 0, 1e-8, hist_sint)
    kl_div = cp.sum(hist_real * cp.log(hist_real / hist_sint)).item()
    cdf_real = cp.cumsum(hist_real)
    cdf_sint = cp.cumsum(hist_sint)
    wd = cp.sum(cp.abs(cdf_real - cdf_sint)).item()
    return kl_div, wd

# =============================================================================
# Execução principal
# =============================================================================
if __name__ == "__main__":
    # Carrega e filtra dados reais
    dados = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
    dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
    dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
    dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek
    dados = dados[dados['dia_da_semana'].between(1, 3)]
    features = ['hora_do_dia', 'PU_longitude', 'PU_latitude', 'DO_longitude', 'DO_latitude']
    dados = dados[features].dropna().sample(frac=0.0001, random_state=42)

    # Split (treino+val) e teste
    train_val_data, test_data = train_test_split(dados, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=0.15, random_state=42)

    # Escalonamento
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data).astype(np.float32)
    val_scaled = scaler.transform(val_data).astype(np.float32)
    test_scaled = scaler.transform(test_data).astype(np.float32)
    hour_mean, hour_std = scaler.mean_[0], scaler.scale_[0]

    # DataLoaders para dados reais
    real_train_dataset = TensorDataset(
        torch.tensor(train_scaled[:, 1:]), torch.tensor(train_scaled[:, 0])
    )
    real_val_dataset = TensorDataset(
        torch.tensor(val_scaled[:, 1:]), torch.tensor(val_scaled[:, 0])
    )
    real_test_dataset = TensorDataset(
        torch.tensor(test_scaled[:, 1:]), torch.tensor(test_scaled[:, 0])
    )
    real_train_loader = DataLoader(real_train_dataset, batch_size=2048, shuffle=True, pin_memory=True)
    real_val_loader = DataLoader(real_val_dataset, batch_size=2048, pin_memory=True)
    real_test_loader = DataLoader(real_test_dataset, batch_size=2048, pin_memory=True)

    # =========================================================================
    # 1. Treino (pré-treinamento) do modelo somente com dados reais
    # =========================================================================
    model_real = DLinear(input_dim=train_scaled.shape[1] - 1, output_dim=1)
    train_with_validation(
        model_real,
        real_train_loader,
        real_val_loader,
        hour_mean,
        hour_std,
        epochs=30,
        lr=0.001,
        patience=5,
        is_finetune=False
    )
    pretrained_metrics_test = evaluate_model(model_real, real_test_loader, hour_mean, hour_std)

    # Salva o modelo pré-treinado
    PRETRAINED_PATH = 'pretrained_dlinear.pth'
    torch.save(model_real.state_dict(), PRETRAINED_PATH)

    # =========================================================================
    # 2. Loop sobre n_clusters = [20, 25, 30, 35]
    # =========================================================================
    n_clusters_list = [20, 25, 30, 35]
    resultados = []

    for nc in n_clusters_list:
        # ---------------------------------------------------------------------
        # Treina GMM para este n_clusters
        # ---------------------------------------------------------------------
        gmm = GaussianMixture(
            num_components=nc,
            covariance_type='full',
            trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1}
        )
        gmm.fit(train_scaled)

        # Gera amostras sintéticas
        num_sint_samples = len(train_data) * 10  # 10x o treino real (exemplo)
        gmm_samples = gmm.model_.sample(num_sint_samples)  # Tensor PyTorch (GPU)

        # Converter para CuPy e calcular divergência
        samples_cp = cp.asarray(gmm_samples.detach().cpu().numpy())
        hour_real_cp = cp.asarray(train_data['hora_do_dia'].values)
        hour_sint_cp = cp.round(samples_cp[:, 0] * hour_std + hour_mean).astype(cp.int32)
        kl_div, wd = calcular_metricas(hour_real_cp, hour_sint_cp)

        # Cria DataLoader sintético
        synth_inputs = torch.tensor(samples_cp[:, 1:].get(), dtype=torch.float32)
        synth_targets = torch.tensor(samples_cp[:, 0].get(), dtype=torch.float32)
        synth_dataset = TensorDataset(synth_inputs, synth_targets)
        synth_loader = DataLoader(synth_dataset, batch_size=2048, shuffle=True, pin_memory=True)

        # ---------------------------------------------------------------------
        # 3. Fine-tuning: carrega modelo pré-treinado e ajusta com dados sintéticos
        # ---------------------------------------------------------------------
        model_finetune = DLinear(input_dim=train_scaled.shape[1] - 1, output_dim=1)
        model_finetune.load_state_dict(torch.load(PRETRAINED_PATH))
        train_with_validation(
            model_finetune,
            synth_loader,
            real_val_loader,
            hour_mean,
            hour_std,
            epochs=30,
            lr=0.0001,
            patience=5,
            is_finetune=True
        )
        finetune_metrics_test = evaluate_model(model_finetune, real_test_loader, hour_mean, hour_std)

        # ---------------------------------------------------------------------
        # 4. "Real + Sintético desde o início" (experimento adicional)
        # ---------------------------------------------------------------------
        real_plus_synth_inputs = np.concatenate((train_scaled[:, 1:], synth_inputs.numpy()), axis=0)
        real_plus_synth_targets = np.concatenate((train_scaled[:, 0], synth_targets.numpy()), axis=0)
        real_plus_synth_dataset = TensorDataset(
            torch.tensor(real_plus_synth_inputs, dtype=torch.float32),
            torch.tensor(real_plus_synth_targets, dtype=torch.float32)
        )
        real_plus_synth_loader = DataLoader(real_plus_synth_dataset, batch_size=2048, shuffle=True, pin_memory=True)

        model_real_plus_synth = DLinear(input_dim=train_scaled.shape[1] - 1, output_dim=1)
        train_with_validation(
            model_real_plus_synth,
            real_plus_synth_loader,
            real_val_loader,
            hour_mean,
            hour_std,
            epochs=30,
            lr=0.001,
            patience=5,
            is_finetune=False
        )
        real_plus_synth_metrics_test = evaluate_model(model_real_plus_synth, real_test_loader, hour_mean, hour_std)

        # ---------------------------------------------------------------------
        # Consolida resultados para este n_clusters
        # ---------------------------------------------------------------------
        resultados.append({
            'n_clusters': nc,
            'kl_div': kl_div,
            'wd': wd,
            'real_mae': pretrained_metrics_test['mae'],
            'real_mse': pretrained_metrics_test['mse'],
            'real_r2': pretrained_metrics_test['r2'],
            'finetune_mae': finetune_metrics_test['mae'],
            'finetune_mse': finetune_metrics_test['mse'],
            'finetune_r2': finetune_metrics_test['r2'],
            'real_synth_init_mae': real_plus_synth_metrics_test['mae'],
            'real_synth_init_mse': real_plus_synth_metrics_test['mse'],
            'real_synth_init_r2': real_plus_synth_metrics_test['r2']
        })

    # =========================================================================
    # 5. Gera DataFrame final e imprime
    # =========================================================================
    df_result = pd.DataFrame(resultados)
    df_result.to_csv('comparacao_n_clusters.csv', index=False)
    print("Resultados consolidados:\n", df_result)

    # =========================================================================
    # 6. Geração de gráficos comparando as métricas x n_clusters
    #    Cada métrica em um gráfico, sem subplot (um plot por métrica).
    # =========================================================================
    # Compararemos Real (pré-treino), Fine-Tune, e Real+Sint (início) nas métricas
    # mae, mse, r2 vs n_clusters

    # Para cada métrica, plotamos as três curvas em função de n_clusters:
    # - real_{metric}
    # - finetune_{metric}
    # - real_synth_init_{metric}

    metrics_to_plot = [('mae', 'MAE'), ('mse', 'MSE'), ('r2', 'R²')]
    for col, title in metrics_to_plot:
        plt.figure()
        plt.plot(df_result['n_clusters'], df_result[f'real_{col}'], marker='o', label='Somente Real')
        plt.plot(df_result['n_clusters'], df_result[f'finetune_{col}'], marker='o', label='Fine-tuning')
        plt.plot(df_result['n_clusters'], df_result[f'real_synth_init_{col}'], marker='o', label='Real+Sint Início')
        plt.xlabel('n_clusters')
        plt.ylabel(title)
        plt.title(f'Comparação de {title} vs n_clusters')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'comparacao_{col}.png')
        plt.close()

    print("Gráficos salvos e comparação concluída.")
