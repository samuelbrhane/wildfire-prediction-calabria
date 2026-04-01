# Model Training and Evaluation

This folder contains the training, hyperparameter tuning and model selection pipelines for all 5 wildfire prediction models. Each model has its own folder with a consistent structure: a shared `model_training.py` and two subfolders for zone-level and regional-level tuning.

---

## Structure

    4_model_training_and_evaluation/
    ├── 01_linear_regression/
    │   ├── model_training.py
    │   ├── zone_level_tuning/
    │   │   ├── results/
    │   │   ├── models/
    │   │   ├── model_selection_results/
    │   │   ├── zone_tuning.py
    │   │   └── zone_model_selection.py
    │   └── regional_level_tuning/
    │       ├── results/
    │       ├── models/
    │       ├── model_selection_results/
    │       ├── regional_tuning.py
    │       └── regional_model_selection.py
    ├── 02_gpr/
    │   └── (same structure as above)
    ├── 03_xgboost/
    │   └── (same structure as above)
    ├── 04_lstm/
    │   └── (same structure as above)
    └── 05_transformer/
        └── (same structure as above)

---

## How It Works

Each model follows the same two-step workflow:

**Step 1 — Tuning:** Run `zone_tuning.py` or `regional_tuning.py` to perform random hyperparameter search. Each trial trains a model, evaluates it on train and validation sets, saves the model file and appends the result to a CSV. If interrupted, the script resumes automatically from the last completed trial.

**Step 2 — Model Selection:** Run `zone_model_selection.py` or `regional_model_selection.py` after tuning completes. It reads the tuning results CSV, selects the top N models by validation performance, evaluates each on the test set and saves predictions, diagnostic plots and a summary CSV.

---

## Models

**`01_linear_regression`**
Linear Regression with fire and climate lag features. Lightweight baseline model. No GPU required.

**`02_gpr`**
Gaussian Process Regression with RBF kernel. Provides uncertainty estimates alongside predictions. Computationally expensive — fewer trials recommended.

**`03_xgboost`**
XGBoost gradient boosted trees with fire and climate lag features. Models saved as `.json` files.

**`04_lstm`**
Multi-layer LSTM with MinMaxScaling and sliding window sequences. Uses early stopping. GPU recommended for full tuning runs.

**`05_transformer`**
Transformer with multi-head attention and time embeddings for month and day-of-week. Most complex model. GPU recommended for full tuning runs.

---

## Shared Utilities

All models import from `3_utils/` for data loading, preprocessing, evaluation, plotting and model selection. See the README in that folder for details.

---

## Output Folders

The following folders are not pushed to the repository. They are created automatically when tuning runs:

- `results/` — tuning results CSV per zone or regional run
- `models/` — saved model files per trial
- `model_selection_results/` — test evaluation CSVs, prediction CSVs, plots and diagnostic summaries

---

## Requirements

- Activate the virtual environment and install dependencies from `requirements.txt` before running any scripts
- Place `zone_sequence_merged.csv` in `1_data/processed/` before running any tuning or selection scripts
- For GPU training set `CUDA_VISIBLE_DEVICES=0` — the scripts detect GPU availability automatically and fall back to CPU if none is found