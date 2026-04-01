# Shared evaluation utility for all models
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X, y, scaler_y=None):
    # Supports Linear Regression, GPR, XGBoost, LSTM and Transformer
    try:
        y_pred, y_std = model.predict(X, return_std=True)
    except TypeError:
        y_pred = model.predict(X)
        y_std = None

    if hasattr(y_pred, "shape") and len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y = scaler_y.inverse_transform(y.reshape(-1, 1)).flatten()
    else:
        y = y.flatten()

    y_pred = np.clip(np.round(y_pred), 0, None)
    y_true = np.clip(np.round(y), 0, None)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_residual = np.mean(y_true - y_pred)
    std_residual = np.std(y_true - y_pred)
    r2 = r2_score(y_true, y_pred)
    exact_match_percentage = (y_true == y_pred).sum() / len(y_true)

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'r2': r2,
        'exact_match_percentage': exact_match_percentage,
    }

    if y_std is not None:
        metrics.update({
            'mean_prediction_std': np.mean(y_std),
            'max_prediction_std': np.max(y_std),
            'min_prediction_std': np.min(y_std),
        })

    return metrics