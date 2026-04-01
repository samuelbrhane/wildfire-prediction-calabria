# Shared constants across all models
import os

# Column names
TARGET_COL = 'Num_Fires'
DATE_COL = 'Date'
ZONE_ID_COL = 'Zone_ID'
FEATURE_COLS = ['Temperature', 'Precipitation', 'Humidity', 'Wind', 'Prev_Num_Fires_Result']

# Training
SEED = 42
EARLY_STOPPING_PATIENCE = 10
MAX_EPOCHS = 50
LOSS_FUNCTION = 'mse'

# Transformer specific
MONTHS_IN_DATA = 8
DAYS_IN_WEEK = 7

# Output folders
RESULTS_DIR = "results"
MODELS_DIR = "models"