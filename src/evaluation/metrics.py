# Computes RMSE, MAE, etc.
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_rmse(y_true, y_pred):
    """Compute Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def compute_mae(y_true, y_pred):
    """Compute Mean Absolute Error"""
    return mean_absolute_error(y_true, y_pred)

def compute_all_metrics(y_true, y_pred):
    """Compute all evaluation metrics"""
    metrics = {
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred)
    }
    return metrics