# Regional-level hyperparameter tuning for LSTM
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from constants import SEED
from tuning_config import LSTM_SEARCH_SPACE as SEARCH_SPACE
from data_loader import load_regional_data
from preprocessing import preprocess_sequences
from evaluation import evaluate_model
from tuning_utils import sample_params, clean_for_python
from model_training import build_and_train_model

# Use GPU if available otherwise fallback to CPU
if tf.config.list_physical_devices('GPU'):
    print("GPU detected — using GPU for training")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("No GPU detected — using CPU for training")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

NUM_TRIALS = 300
GROUP_NAME = "regional"

results_csv_path = os.path.join(RESULTS_DIR, f"{GROUP_NAME}_lstm_results.csv")
model_dir = os.path.join(MODELS_DIR, GROUP_NAME)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Resume from last completed trial if exists
completed_trials = 0
if os.path.exists(results_csv_path):
    completed_trials = len(pd.read_csv(results_csv_path))
    print(f"Found {completed_trials} completed trials — resuming...")

print("Loading regional data (all 8 zones aggregated)")
df = load_regional_data()
print(f"Total rows: {len(df)}")

for trial in range(completed_trials, NUM_TRIALS):
    print(f"Trial {trial + 1}/{NUM_TRIALS}")
    try:
        params = sample_params(SEARCH_SPACE)
        X_train, y_train, X_val, y_val, X_test, y_test, df_test, scaler_y = preprocess_sequences(df.copy(), params)

        model, history = build_and_train_model(X_train, y_train, X_val, y_val, params)

        train_metrics = evaluate_model(model, X_train, y_train, scaler_y=scaler_y)
        val_metrics = evaluate_model(model, X_val, y_val, scaler_y=scaler_y)

        model_filename = f"trial_{trial + 1:03d}.keras"
        model_path = os.path.join(model_dir, model_filename)
        model.save(model_path)

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

    except tf.errors.ResourceExhaustedError:
        print(f"Skipping trial {trial + 1} — out of memory")
        continue
    except Exception as e:
        print(f"Skipping trial {trial + 1} due to error: {e}")
        continue
    finally:
        K.clear_session()

print(f"\nTuning complete for regional LSTM model — results saved to {results_csv_path}")