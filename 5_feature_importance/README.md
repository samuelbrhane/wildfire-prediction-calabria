# Feature Importance

This folder contains feature importance analysis for the best selected models from the wildfire prediction pipeline. Analysis is done for XGBoost at the zone level and GPR at the regional level.

---

## How Model Selection Works

Each tuning run produces up to 300 trials. The model selection step evaluates the top 10 candidates on the test set ranked by validation performance (exact match percentage, R², RMSE and MAE). From these 10, the final model is **manually selected** based on a balance of metrics and visual inspection of the predicted vs observed plots and error distribution. The selected trial filename is then hardcoded in the notebooks here.

---

## Files

**`xgboost_zone_importance.ipynb`**
Loads the manually selected XGBoost model for each of the 8 zones and computes feature importance using XGBoost's built-in gain metric. Produces individual plots per zone and a combined 2×4 subplot for comparison across all zones.

**`gpr_regional_importance.ipynb`**
Loads the manually selected regional GPR model and computes feature sensitivity using ARD kernel length scales alongside SHAP values. Produces a side-by-side plot of length scales and SHAP summary.

**`plots/`**
All output plots are saved here.

---

## Updating Selected Models

If you retrain the models and select different trials, update the following in each notebook:

In `xgboost_zone_importance.ipynb`:
```python
zone_trials = {
    1: "trial_093.json",
    2: "trial_051.json",
    ...
}
```

In `gpr_regional_importance.ipynb`:
```python
selected_filename = "trial_001.pkl"
```

---

## Requirements

- Tuning and model selection must be completed for `03_xgboost` zone level and `02_gpr` regional level before running these notebooks
- Model selection results CSVs must exist in the respective `model_selection_results/` folders
- `zone_sequence_merged.csv` must be present in `1_data/processed/`