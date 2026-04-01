# Zone-level model selection and evaluation for Transformer
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loader import load_zone_data
from preprocessing import preprocess_sequences_with_time
from model_selection import evaluate_top_models
from evaluation import evaluate_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
SELECTION_DIR = os.path.join(CURRENT_DIR, "model_selection_results")


def get_params(row):
    # Extract Transformer hyperparameters from result row
    return {
        'batch_size': int(row['batch_size']),
        'learning_rate': float(row['learning_rate']),
        'dropout_rate': float(row['dropout_rate']),
        'lag_days': int(row['lag_days']),
        'sequence_length': int(row['sequence_length']),
        'd_model': int(row['d_model']),
        'num_heads': int(row['num_heads']),
        'ff_dim': int(row['ff_dim']),
        'num_layers': int(row['num_layers']),
        'month_embed_dim': int(row['month_embed_dim']),
        'dow_embed_dim': int(row['dow_embed_dim']),
        'optimizer': str(row['optimizer'])
    }


def make_preprocess_fn(zone_id, params):
    # Returns preprocessed sequence data with time features for a given zone
    df = load_zone_data(zone_id)
    return preprocess_sequences_with_time(df.copy(), params)


def load_model_fn(model_path):
    # Load Keras Transformer model with custom objects
    return load_model(model_path, custom_objects={"mse": MeanSquaredError()})


def predict_fn(model, preprocess_result):
    # Predict and inverse transform for Transformer zone model
    _, _, _, _, _, _, _, _, X_test, y_test, month_test, dow_test, df_test, scaler_y = preprocess_result
    y_pred_scaled = model.predict([X_test, month_test, dow_test])
    y_pred = np.round(np.clip(
        scaler_y.inverse_transform(y_pred_scaled).flatten(), 0, None
    ))
    y_true = np.round(np.clip(
        scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten(), 0, None
    ))
    residuals = y_true - y_pred
    test_dates = df_test["Date"].reset_index(drop=True)
    metrics = evaluate_model(model, [X_test, month_test, dow_test], y_test, scaler_y=scaler_y)
    return y_pred, y_true, residuals, test_dates, metrics


for zone_id in range(1, 2):
    print(f"\n=== Model Selection for Zone {zone_id} ===")
    result_file = os.path.join(RESULTS_DIR, f"zone_{zone_id}_transformer_results.csv")

    evaluate_top_models(
        group_name=f"zone_{zone_id}",
        result_file=result_file,
        save_dir=SELECTION_DIR,
        model_type="transformer",
        preprocess_fn=lambda params, z=zone_id: make_preprocess_fn(z, params),
        load_model_fn=load_model_fn,
        predict_fn=predict_fn,
        get_params_fn=get_params,
        zone_id=zone_id,
        top_n=10
    )

print("\nAll Transformer zone model selection complete.")