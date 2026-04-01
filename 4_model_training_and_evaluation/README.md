# Utils

This folder contains all shared utilities used across all models in `4_model_training_and_evaluation/`. Every model imports from here — nothing in this folder is model-specific.

---

## Files

**`constants.py`**
Shared constants used across all models: column names, training hyperparameters (seed, epochs, early stopping patience, loss function), and Transformer-specific time embedding dimensions.

**`tuning_config.py`**
Hyperparameter search spaces for all 5 models: Linear Regression, GPR, XGBoost, LSTM and Transformer. Each model's tuning script imports its own search space from here.

**`data_loader.py`**
Two functions for loading the merged dataset: `load_zone_data(zone_id)` loads data for a single zone, and `load_regional_data()` loads and aggregates data across all 8 zones into a regional series. The data path is resolved automatically relative to this file so it works regardless of where scripts are called from.

**`preprocessing.py`**
Three preprocessing functions covering all 5 models: `preprocess_lag_features(df, params)` is used by Linear Regression, GPR and XGBoost and creates fire and climate lag features then splits into train/val/test. `preprocess_sequences(df, params)` is used by LSTM and applies MinMaxScaler and creates sliding window sequences. `preprocess_sequences_with_time(df, params)` is used by Transformer and does the same as LSTM but also extracts month and day-of-week time features for embeddings. All functions use a 65/15/20 train/val/test split.

**`evaluation.py`**
Single `evaluate_model(model, X, y, scaler_y=None)` function that works across all model types. Handles GPR uncertainty estimates, LSTM and Transformer inverse scaling, and computes RMSE, MAE, R², mean residual, std residual and exact match percentage.

**`tuning_utils.py`**
Two helper functions used by all tuning scripts: `sample_params(search_space)` randomly samples one value per hyperparameter from the search space handling both integer and float ranges, and `clean_for_python(obj)` recursively converts NumPy types to native Python types for safe CSV serialization.

**`plots.py`**
Single `save_summary_and_plots(...)` function that generates and saves two plots and a diagnostics text file for any evaluated model: a predicted vs observed time series plot and a prediction error distribution plot.

**`model_selection.py`**
Single `evaluate_top_models(...)` function that reads tuning results, selects the top N models by validation performance, evaluates them on the test set, and saves predictions, plots and a summary CSV. Used by all model selection scripts across all 5 models.

---

## How to Import

All model scripts add this folder to the path at runtime:

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '3_utils'))

The exact number of `..` depends on how deep the calling script is relative to the repo root.