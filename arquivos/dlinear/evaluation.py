import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calcula R², MAE e RMSE."""
    # Garante que os inputs sejam arrays numpy planos
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Evita divisão por zero ou R² indefinido se y_true for constante
    if np.all(y_true == y_true[0]):
         r2 = np.nan # Ou 0, dependendo da interpretação desejada
    else:
        r2 = r2_score(y_true, y_pred)

    metrics = {
        'R²': r2,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics