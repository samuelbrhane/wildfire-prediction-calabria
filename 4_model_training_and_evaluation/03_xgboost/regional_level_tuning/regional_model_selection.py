# Regional-level model selection and evaluation for XGBoost
import os
import sys
import numpy as np
from xgboost import XGBRegressor

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
    # Extract XGBoost hyperparameters from result row
    return {
        'fire_lag': int(row['fire_lag']),
        'climate_lag': int(row['climate_lag']),
        'n_estimators': int(row['n_estimators']),
        'max_depth': int(row['max_depth']),
        'learning_rate': float(row['learning_rate']),
        'subsample': float(row['subsample']),
        'colsample_bytree': float(row['colsample_bytree']),
        'gamma': float(row['gamma']),
        'reg_alpha': float(row['reg_alpha']),
        'reg_lambda': float(row['reg_lambda'])
    }


def make_preprocess_fn(params):
    # Returns preprocessed regional data for given params
    df = load_regional_data()
    return preprocess_lag_features(df.copy(), params)


def load_model_fn(model_path):
    # Load XGBoost model from JSON format
    model = XGBRegressor()
    model.load_model(model_path)
    return model


def predict_fn(model, preprocess_result):
    # Predict and compute residuals for XGBoost regional model
    _, _, _, _, X_test, y_test, df_test = preprocess_result
    y_pred = np.round(np.clip(model.predict(X_test), 0, None))
    y_true = np.round(np.clip(y_test, 0, None))
    residuals = y_true - y_pred
    test_dates = df_test["Date"].reset_index(drop=True)
    metrics = evaluate_model(model, X_test, y_test)
    return y_pred, y_true, residuals, test_dates, metrics


result_file = os.path.join(RESULTS_DIR, "regional_xgboost_results.csv")

print("\n=== Model Selection for Regional XGBoost ===")
evaluate_top_models(
    group_name="regional",
    result_file=result_file,
    save_dir=SELECTION_DIR,
    model_type="xgboost",
    preprocess_fn=make_preprocess_fn,
    load_model_fn=load_model_fn,
    predict_fn=predict_fn,
    get_params_fn=get_params,
    zone_id=None,
    top_n=10
)

print("\nXGBoost regional model selection complete.")