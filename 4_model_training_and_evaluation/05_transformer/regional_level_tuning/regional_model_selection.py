# Regional-level model selection and evaluation for Transformer
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from data_loader import load_regional_data
from preprocessing import preprocess_sequences_with_time
from evaluation import evaluate_model
from model_selection import evaluate_top_models

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


def make_preprocess_fn(params):
    # Returns preprocessed regional sequence data with time features for given params
    df = load_regional_data()
    return preprocess_sequences_with_time(df.copy(), params)


def load_model_fn(model_path):
    # Load Keras Transformer model with custom objects
    return load_model(model_path, custom_objects={"mse": MeanSquaredError()})


def predict_fn(model, preprocess_result):
    # Predict and inverse transform for Transformer regional model
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


result_file = os.path.join(RESULTS_DIR, "regional_transformer_results.csv")

print("\n=== Model Selection for Regional Transformer ===")
evaluate_top_models(
    group_name="regional",
    result_file=result_file,
    save_dir=SELECTION_DIR,
    model_type="transformer",
    preprocess_fn=make_preprocess_fn,
    load_model_fn=load_model_fn,
    predict_fn=predict_fn,
    get_params_fn=get_params,
    zone_id=None,
    top_n=10
)

print("\nTransformer regional model selection complete.")
