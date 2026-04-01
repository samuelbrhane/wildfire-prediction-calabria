# Regional-level model selection and evaluation for Linear Regression
import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loader import load_regional_data
from preprocessing import preprocess_lag_features
from model_selection import evaluate_top_models

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
SELECTION_DIR = os.path.join(CURRENT_DIR, "model_selection_results")


def get_params(row):
    # Extract Linear Regression hyperparameters from result row
    return {
        'fire_lag': int(row['fire_lag']),
        'climate_lag': int(row['climate_lag']),
        'fit_intercept': bool(row['fit_intercept'])
    }


def make_preprocess_fn(params):
    # Returns preprocessed regional data for given params
    df = load_regional_data()
    return preprocess_lag_features(df.copy(), params)


def predict_fn(model, X_test, y_test, df_test, scaler_y=None):
    # Predict and compute residuals for Linear Regression regional model
    y_pred = np.round(np.clip(model.predict(X_test), 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred
    test_dates = df_test["Date"].reset_index(drop=True)
    return y_pred, y_true, residuals, test_dates


result_file = os.path.join(RESULTS_DIR, "regional_linear_results.csv")

print("\n=== Model Selection for Regional Linear Regression ===")
evaluate_top_models(
    group_name="regional",
    result_file=result_file,
    save_dir=SELECTION_DIR,
    model_type="linear",
    preprocess_fn=make_preprocess_fn,
    load_model_fn=joblib.load,
    predict_fn=predict_fn,
    get_params_fn=get_params,
    zone_id=None
)
