# Zone-level model selection and evaluation for Linear Regression
import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loader import load_zone_data
from preprocessing import preprocess_lag_features
from model_selection import evaluate_top_models
from evaluation import evaluate_model

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


def make_preprocess_fn(zone_id, params):
    # Returns preprocessed data for a given zone and params
    df = load_zone_data(zone_id)
    return preprocess_lag_features(df.copy(), params)


def predict_fn(model, preprocess_result):
    # Predict and compute residuals for Linear Regression
    _, _, _, _, X_test, y_test, df_test = preprocess_result
    y_pred = np.round(np.clip(model.predict(X_test), 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred
    test_dates = df_test["Date"].reset_index(drop=True)
    metrics = evaluate_model(model, X_test, y_test)
    return y_pred, y_true, residuals, test_dates, metrics


for zone_id in range(1, 9):
    print(f"\n=== Model Selection for Zone {zone_id} ===")
    result_file = os.path.join(RESULTS_DIR, f"zone_{zone_id}_linear_results.csv")

    evaluate_top_models(
        group_name=f"zone_{zone_id}",
        result_file=result_file,
        save_dir=SELECTION_DIR,
        model_type="linear",
        preprocess_fn=lambda params, z=zone_id: make_preprocess_fn(z, params),
        load_model_fn=joblib.load,
        predict_fn=predict_fn,
        get_params_fn=get_params,
        zone_id=zone_id
    )