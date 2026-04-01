# Zone-level hyperparameter tuning for Linear Regression
import os
import sys
import random
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))

from constants import SEED
from tuning_config import LINEAR_REGRESSION_SEARCH_SPACE as SEARCH_SPACE
from data_loader import load_zone_data
from preprocessing import preprocess_lag_features
from evaluation import evaluate_model
from tuning_utils import sample_params, clean_for_python
from model_training import build_and_train_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

random.seed(SEED)
np.random.seed(SEED)

NUM_TRIALS = 20

for zone_id in range(1, 9):
    print(f"\n=== Tuning Linear Regression for Zone {zone_id} ===")
    try:
        df = load_zone_data(zone_id)
    except Exception as e:
        print(f"Skipping Zone {zone_id} due to error: {e}")
        continue

    model_dir = os.path.join(MODELS_DIR, f"zone_{zone_id}")
    os.makedirs(model_dir, exist_ok=True)
    results = []

    for trial in range(NUM_TRIALS):
        print(f"Trial {trial + 1}/{NUM_TRIALS}")
        try:
            params = sample_params(SEARCH_SPACE)
            X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_lag_features(df.copy(), params)

            model = build_and_train_model(X_train, y_train, params)

            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)

            model_filename = f"trial_{trial + 1:03d}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)

            trial_result = {
                **params,
                "model_file": model_path,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }
            results.append(trial_result)

        except Exception as e:
            print(f"Skipping trial {trial + 1} due to error: {e}")
            continue

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_py_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_linear_results.py")
    results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_linear_results.csv")

    with open(results_py_path, 'w') as f:
        f.write("results = [\n")
        for res in clean_for_python(results):
            f.write(f"    {res},\n")
        f.write("]\n")

    pd.DataFrame(results).to_csv(results_csv_path, index=False)
    print(f"Zone {zone_id} complete — results saved to {results_csv_path}")