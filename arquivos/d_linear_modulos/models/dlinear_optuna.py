import optuna
import torch

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
    n_trials: int = 50,
    seed: int | None = None,
):
    """Optimize DLinear hyperparameters using Optuna.
    ...
    """
    
    # 1. Defina o dispositivo aqui, para que ele esteja disponível para cada "trial"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando otimização no dispositivo: {device}")
    device = 'cuda:0'

    def objective(trial: optuna.Trial):
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch = trial.suggest_int("batch_size", 32, 128, step=32)
        epochs = trial.suggest_int("epochs", 50, 400, step=50)

        model = DLinearModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len).to(device)
        
        # 2. Passe o 'device' para a função de treino
        train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch
           # device=device
        )

        # Após o treino, o modelo já está no 'device' correto.
        # Agora, para fazer a predição, os dados também precisam estar lá.
        with torch.no_grad():
            # 3. Mova os dados de validação para o mesmo dispositivo do modelo
            pred_tensor = model(X_val.to(device))
            
            # Mova os resultados de volta para a CPU para usar com NumPy/Scikit-learn
            pred = pred_tensor.squeeze().cpu().numpy()
            true = y_val.squeeze().cpu().numpy()

        metrics = compute_metrics(true, pred)
        r2 = metrics["R²"]
        smape_val = smape(true, pred)

        return r2, smape_val

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
    
    # É uma boa prática rodar jobs de otimização na GPU em uma única thread (n_jobs=1)
    # para evitar conflitos de alocação de memória na VRAM.
    study.optimize(objective, n_trials=n_trials, n_jobs=1) 
    
    return study