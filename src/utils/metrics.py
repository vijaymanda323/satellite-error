import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, and R2 for each target dimension separately.
    y_true: (n_samples, 2)
    y_pred: (n_samples, 2)
    """
    metrics = {
        "clock": {},
        "orbit": {}
    }
    
    for i, target in enumerate(["clock", "orbit"]):
        metrics[target]["MAE"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[target]["RMSE"] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[target]["R2"] = r2_score(y_true[:, i], y_pred[:, i])
        
    return metrics
