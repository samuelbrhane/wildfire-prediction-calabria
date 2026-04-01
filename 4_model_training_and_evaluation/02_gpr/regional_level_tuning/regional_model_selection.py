# Regional-level model selection and evaluation for GPR
import os
import sys
import numpy as np
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loader import load_regional_data
from preprocessing import preprocess_lag_features
from model_selection import evaluate_top_models
from evaluation import evaluate_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
SELECTION_DIR = os.path.join(CURRENT_DIR, "model_selection_results")


def get_params(row):
    # Extract GPR hyperparameters from result row
    return {
        'fire_lag': int(row['fire_lag']),
        'climate_lag': int(row['climate_lag']),
        'constant_value': float(row['constant_value']),
        'length_scale': float(row['length_scale']),
        'alpha': float(row['alpha'])
    }


def make_preprocess_fn(params):
    # Returns preprocessed regional data for given params
    df = load_regional_data()
    return preprocess_lag_features(df.copy(), params)


def predict_fn(model, preprocess_result):
    # Predict with uncertainty estimates for GPR regional model
    _, _, _, _, X_test, y_test, df_test = preprocess_result
    y_pred_raw, _ = model.predict(X_test, return_std=True)
    y_pred = np.round(np.clip(y_pred_raw, 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred
    test_dates = df_test["Date"].reset_index(drop=True)
    metrics = evaluate_model(model, X_test, y_test)
    return y_pred, y_true, residuals, test_dates, metrics


result_file = os.path.join(RESULTS_DIR, "regional_gpr_results.csv")

print("\n=== Model Selection for Regional GPR ===")
evaluate_top_models(
    group_name="regional",
    result_file=result_file,
    save_dir=SELECTION_DIR,
    model_type="gpr",
    preprocess_fn=make_preprocess_fn,
    load_model_fn=joblib.load,
    predict_fn=predict_fn,
    get_params_fn=get_params,
    zone_id=None,
    top_n=10
)

print("\nGPR regional model selection complete.")