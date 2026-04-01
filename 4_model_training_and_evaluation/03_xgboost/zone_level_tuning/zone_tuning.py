# Zone-level hyperparameter tuning for XGBoost
import os
import sys
import random
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from constants import SEED
from tuning_config import XGBOOST_SEARCH_SPACE as SEARCH_SPACE
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

NUM_TRIALS = 300

for zone_id in range(1, 9):
    print(f"\n=== Tuning XGBoost for Zone {zone_id} ===")

    results_csv_path = os.path.join(RESULTS_DIR, f"zone_{zone_id}_xgboost_results.csv")
    model_dir = os.path.join(MODELS_DIR, f"zone_{zone_id}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resume from last completed trial if exists
    completed_trials = 0
    if os.path.exists(results_csv_path):
        completed_trials = len(pd.read_csv(results_csv_path))
        print(f"Found {completed_trials} completed trials — resuming...")

    try:
        df = load_zone_data(zone_id)
        print(f"Loaded data for Zone {zone_id}: {len(df)} rows")
    except Exception as e:
        print(f"Skipping Zone {zone_id} due to error: {e}")
        continue

    for trial in range(completed_trials, NUM_TRIALS):
        print(f"Trial {trial + 1}/{NUM_TRIALS}")
        try:
            params = sample_params(SEARCH_SPACE)
            X_train, y_train, X_val, y_val, X_test, y_test, _ = preprocess_lag_features(df.copy(), params)

            model = build_and_train_model(X_train, y_train, params, seed=SEED, X_val=X_val, y_val=y_val)

            train_metrics = evaluate_model(model, X_train, y_train)
            val_metrics = evaluate_model(model, X_val, y_val)

            model_filename = f"trial_{trial + 1:03d}.json"
            model_path = os.path.join(model_dir, model_filename)
            model.save_model(model_path)

            trial_result = clean_for_python({
                **params,
                "model_file": model_path,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

            # Append result immediately after each trial
            pd.DataFrame([trial_result]).to_csv(
                results_csv_path,
                mode='a',
                header=not os.path.exists(results_csv_path),
                index=False
            )

        except Exception as e:
            print(f"Skipping trial {trial + 1} due to error: {e}")
            continue

    print(f"Zone {zone_id} complete — results saved to {results_csv_path}")

print("\nAll XGBoost zone tuning complete.")