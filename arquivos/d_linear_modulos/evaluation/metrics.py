import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return np.nan if np.isinf(mape) or mape > 1e9 else mape

def compute_metrics(true, pred):
    return {
        "R²"  : r2_score(true, pred) if len(true) > 1 else np.nan,
        "MAE" : mean_absolute_error(true, pred),
        "RMSE": np.sqrt(mean_squared_error(true, pred)),
        "MAPE": mean_absolute_percentage_error(true, pred),
    }
