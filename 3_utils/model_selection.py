# Shared model selection and evaluation utility for all models
import os
import pandas as pd
from plots import save_summary_and_plots
from evaluation import evaluate_model


def evaluate_top_models(
    group_name,
    result_file,
    save_dir,
    model_type,
    preprocess_fn,
    load_model_fn,
    predict_fn,
    get_params_fn,
    zone_id=None,
    top_n=10
):
    # Evaluates top tuning candidates on test set for zone and regional models
    if not os.path.exists(result_file):
        print(f"Tuning results not found at {result_file}. Skipping.")
        return

    results_df = pd.read_csv(result_file)
    top_candidates = results_df.sort_values(
        by=[
            "val_exact_match_percentage",
            "val_r2",
            "val_rmse",
            "val_mae",
            "train_exact_match_percentage"
        ],
        ascending=[False, False, True, True, False]
    ).head(top_n)

    os.makedirs(save_dir, exist_ok=True)
    evaluated_models = []

    for i, model_row in enumerate(top_candidates.to_dict("records")):
        model_path = model_row["model_file"]
        model_filename = os.path.basename(model_path)
        print(f"\nEvaluating model {i + 1}: {model_filename}")

        params = get_params_fn(model_row)

        try:
            model = load_model_fn(model_path)
        except Exception as e:
            print(f"Failed to load model {model_filename}: {e}")
            continue

        # Pass full preprocess result to predict_fn so each model unpacks what it needs
        preprocess_result = preprocess_fn(params)

        y_pred, y_true, residuals, test_dates, metrics = predict_fn(
            model, preprocess_result
        )

        print(f"Test metrics for {model_filename}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        output_subdir = f"zone_{zone_id}" if zone_id is not None else "regional"
        output_dir = os.path.join(save_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        base_name = model_filename.replace('.pkl', '').replace('.keras', '')

        pred_df = pd.DataFrame({
            "Date": test_dates,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": residuals
        })
        pred_df.to_csv(os.path.join(output_dir, f"test_predictions_{base_name}.csv"), index=False)

        save_summary_and_plots(
            model_filename=model_filename,
            output_dir=output_dir,
            base_name=base_name,
            y_true=y_true,
            y_pred=y_pred,
            residuals=residuals,
            metrics=metrics,
            test_dates=test_dates,
            zone_id=zone_id
        )

        evaluated_models.append({
            "model_filename": model_filename,
            "model_path": model_path,
            **params,
            **metrics
        })

    if evaluated_models:
        evaluated_df = pd.DataFrame(evaluated_models)
        result_name = f"{group_name}_{model_type}_top_models_test_eval.csv"
        evaluated_df.to_csv(os.path.join(save_dir, result_name), index=False)
        print(f"\nSaved test evaluation results to {os.path.join(save_dir, result_name)}")