# Shared plotting and diagnostics utility for all models
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


def save_summary_and_plots(
    model_filename,
    output_dir,
    base_name,
    y_true,
    y_pred,
    residuals,
    metrics,
    test_dates,
    zone_id=None,
    y_std=None
):
    # Saves diagnostics summary and performance plots for zone and regional models
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"summary_test_diagnostics_{base_name}.txt")
    with open(summary_path, "w") as f:
        f.write(f"Model File: {model_filename}\n")
        if zone_id is not None:
            f.write(f"Zone ID: {zone_id}\n")
        f.write(f"Number of Test Samples (days evaluated): {len(y_true)}\n\n")
        f.write("Model Performance on Test Data:\n")
        f.write(f" - RMSE: {metrics['rmse']:.2f} fires/day\n")
        f.write(f" - MAE: {metrics['mae']:.2f} fires/day\n")
        f.write(f" - Mean Residual: {metrics['mean_residual']:.2f}\n")
        f.write(f" - Std Residual: {metrics['std_residual']:.2f}\n")
        f.write(f" - R² Score: {metrics['r2']:.3f}\n")
        f.write(f" - Exact Match %: {metrics['exact_match_percentage']:.2%}\n")
        if y_std is not None and 'mean_prediction_std' in metrics:
            f.write(f" - Mean Prediction Std: {metrics['mean_prediction_std']:.2f}\n")
            f.write(f" - Max Prediction Std: {metrics['max_prediction_std']:.2f}\n")
    print(f"Saved summary: {summary_path}")

    x = np.arange(len(y_true))
    date_labels = test_dates.astype(str)
    tick_step = max(1, len(date_labels) // 25)
    xticks = np.arange(0, len(date_labels), tick_step)
    zero_offset = 0.02
    y_true_plot = np.where(y_true == 0, zero_offset, y_true)
    y_pred_plot = np.where(y_pred == 0, zero_offset, y_pred)

    # Predicted vs Observed
    plt.figure(figsize=(9, 5))
    plt.plot(x, y_true_plot, label='Observed Fire Counts', color='blue', linewidth=0.6)
    plt.plot(x, y_pred_plot, label='Predicted Fire Counts', color='red', linewidth=0.6)
    plt.xticks(ticks=xticks, labels=date_labels[::tick_step], rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    y_interval = 1 if zone_id is not None else 5
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(y_interval))
    plt.ylim(bottom=0)
    plt.grid(True, linestyle='--', linewidth=0.4)
    plt.title('Predicted vs. Observed Fires', fontsize=10)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Number of Fires', fontsize=8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"test_predicted_vs_observed_{base_name}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Error Distribution
    plt.figure(figsize=(9, 5))
    min_val = int(np.floor(residuals.min()))
    max_val = int(np.ceil(residuals.max()))
    bin_step = 2 if zone_id is None else 1
    bin_edges = np.arange(min_val - 0.5, max_val + bin_step + 0.5, bin_step)

    histplot = sns.histplot(residuals, bins=bin_edges, color='orange', edgecolor='k', alpha=0.75)
    for patch in histplot.patches:
        height = patch.get_height()
        if height > 0:
            x_text = patch.get_x() + patch.get_width() / 2
            plt.text(x_text, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=6)

    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
    plt.title('Distribution of Daily Prediction Errors on Test Set', fontsize=10)
    plt.xlabel('Prediction Error (Observed - Predicted Fire Count)', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(bin_step))
    plt.grid(True)
    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, f"test_prediction_error_distribution_{base_name}.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"Saved error plot: {error_plot_path}")