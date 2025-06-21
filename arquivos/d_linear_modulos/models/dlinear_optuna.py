import optuna
import torch

from .dlinear import DLinearModel, train_model
from utils.helpers import smape
from evaluation.metrics import compute_metrics

import optuna
import torch
from optuna.exceptions import TrialPruned

# Supondo que seus imports estejam corretos
from .dlinear import DLinearModel, train_model
from utils.helpers import smape
from evaluation.metrics import compute_metrics


def optimize_dlinear(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    output_dim: int,
    seq_len: int,
    n_trials: int = 100,
    seed: int | None = None,
):
    """Otimiza os hiperparâmetros do DLinear usando Optuna com poda e espaço de busca aprimorado."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando otimização no dispositivo: {device}")

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # --- [NOVO] Espaço de Busca Aprimorado ---
        # 1. Parâmetros do otimizador e treinamento
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
        
        # 2. Parâmetros da arquitetura do modelo
        kernel_size = trial.suggest_int("kernel_size", 3, 31, step=2) # Kernels ímpares
        
        # 3. Parâmetros de controle do treinamento
        epochs = trial.suggest_int("epochs", 50, 200, step=50)
        patience = trial.suggest_int("patience", 5, 20)

        # Criação do modelo com parâmetros dinâmicos
        model = DLinearModel(
            input_dim=input_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            kernel_size=kernel_size
        ).to(device)
        
        try:
            # --- [IMPORTANTE] Modificação para suportar Pruning ---
            # O 'trial' do Optuna é passado para a função de treino
            train_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
                learning_rate=lr,
                batch_size=batch_size,
                patience=patience,
                weight_decay=weight_decay, # [NOVO] Passando weight_decay
                trial=trial                # [NOVO] Passando o objeto trial
            )

        except TrialPruned:
            # Se o Optuna podar o trial, repassamos a exceção
            raise

        except Exception as e:
            # Lidar com outros erros inesperados (ex: gradientes explodindo)
            print(f"Trial falhou com uma exceção: {e}")
            # Retorne valores muito ruins para que o Optuna descarte este trial
            return -1.0, 1e9 # R² ruim, SMAPE ruim

        # Avaliação final do modelo treinado
        model.eval()
        with torch.no_grad():
            pred_tensor = model(X_val.to(device))
            pred = pred_tensor.squeeze().cpu().numpy()
            true = y_val.squeeze().cpu().numpy()

        metrics = compute_metrics(true, pred)
        r2 = metrics.get("R²", -1.0) # Usar .get para evitar erro se a métrica falhar
     #   mae = metrics.get("MAE", 1e9)
        
        # [MODIFICADO] Retorna as métricas que queremos otimizar
        return r2

    # --- [NOVO] Configuração do Estudo com Pruner ---
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10) # Ignora as 10 primeiras épocas
    
    study = optuna.create_study(
        directions=["maximize"], # Maximizar R² e Minimizar MAE
        sampler=sampler,
        pruner=pruner
    )
    
    study.optimize(objective, n_trials=n_trials, n_jobs=1) 
    
    return study