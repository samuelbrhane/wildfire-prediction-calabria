# Top-level script to run all 5 model selection pipelines (zone + regional) sequentially
import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "4_model_training_and_evaluation")

SELECTION_SCRIPTS = [
    # (stage_name, script_path)
    ("Linear Regression - Zone",     "01_linear_regression/zone_level_tuning/zone_model_selection.py"),
    ("Linear Regression - Regional", "01_linear_regression/regional_level_tuning/regional_model_selection.py"),
    ("GPR - Zone",                   "02_gpr/zone_level_tuning/zone_model_selection.py"),
    ("GPR - Regional",               "02_gpr/regional_level_tuning/regional_model_selection.py"),
    ("XGBoost - Zone",               "03_xgboost/zone_level_tuning/zone_model_selection.py"),
    ("XGBoost - Regional",           "03_xgboost/regional_level_tuning/regional_model_selection.py"),
    ("LSTM - Zone",                  "04_lstm/zone_level_tuning/zone_model_selection.py"),
    ("LSTM - Regional",              "04_lstm/regional_level_tuning/regional_model_selection.py"),
    ("Transformer - Zone",           "05_transformer/zone_level_tuning/zone_model_selection.py"),
    ("Transformer - Regional",       "05_transformer/regional_level_tuning/regional_model_selection.py"),
]

PROGRESS_FILE = os.path.join(BASE_DIR, "model_selection_progress.txt")


def load_completed():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        return set(line.strip() for line in f.readlines() if line.strip())


def mark_completed(name):
    with open(PROGRESS_FILE, "a") as f:
        f.write(name + "\n")


def main():
    completed = load_completed()
    print(f"=== Wildfire Prediction — Full Model Selection Pipeline ===")
    print(f"Found {len(completed)} already completed stages\n")

    for name, relative_path in SELECTION_SCRIPTS:
        if name in completed:
            print(f"[SKIP] {name} — already completed")
            continue

        script_path = os.path.join(MODEL_DIR, relative_path)
        print(f"\n{'='*60}")
        print(f"[START] {name}")
        print(f"Script: {script_path}")
        print(f"{'='*60}")

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path)
        )

        if result.returncode == 0:
            mark_completed(name)
            print(f"\n[DONE] {name} completed successfully")
        else:
            print(f"\n[ERROR] {name} exited with code {result.returncode}")
            print("Stopping pipeline — fix the error and rerun to resume")
            sys.exit(result.returncode)

    print(f"\n{'='*60}")
    print("All 10 model selection stages complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()