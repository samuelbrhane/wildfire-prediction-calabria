# Shared preprocessing utilities for all models
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from constants import FEATURE_COLS, TARGET_COL, DATE_COL


def _split_data(df):
    """Split dataframe into train, val and test sets using 65/15/20 ratio."""
    split_1 = int(len(df) * 0.65)
    split_2 = int(len(df) * 0.80)
    df_train = df[:split_1].dropna().reset_index(drop=True)
    df_val = df[split_1:split_2].dropna().reset_index(drop=True)
    df_test = df[split_2:].dropna().reset_index(drop=True)
    return df_train, df_val, df_test


def _create_sequences(X, y, seq_len):
    """Create sliding window sequences from feature and target arrays."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def _create_sequences_with_time(X, y, month, dow, seq_len):
    """Create sliding window sequences including month and day-of-week time features."""
    X_seq, y_seq, month_seq, dow_seq = [], [], [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
        month_seq.append(month[i:i + seq_len])
        dow_seq.append(dow[i:i + seq_len])
    return np.array(X_seq), np.array(y_seq), np.array(month_seq), np.array(dow_seq)


def preprocess_lag_features(df, params):
    """Used by Linear Regression, GPR and XGBoost for both zone and regional models"""
    fire_lag = params['fire_lag']
    climate_lag = params['climate_lag']
    climate_features = ['Temperature', 'Precipitation', 'Humidity', 'Wind']

    for lag in range(1, fire_lag + 1):
        df[f'Num_Fires_lag_{lag}'] = df[TARGET_COL].shift(lag)

    for col in climate_features:
        for lag in range(1, climate_lag + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    lagged_fire = [f'Num_Fires_lag_{i}' for i in range(1, fire_lag + 1)]
    lagged_climate = [f'{col}_lag_{i}' for col in climate_features for i in range(1, climate_lag + 1)]
    feature_cols = climate_features + lagged_fire + lagged_climate

    df_train, df_val, df_test = _split_data(df)

    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[feature_cols].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET_COL].values

    return X_train, y_train, X_val, y_val, X_test, y_test, df_test


def preprocess_sequences(df, params):
    """Used by LSTM for both zone and regional models"""
    df['Prev_Num_Fires_Result'] = df[TARGET_COL].shift(params['lag_days'])

    df_train, df_val, df_test = _split_data(df)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    seq_len = params['sequence_length']
    X_train_seq, y_train_seq = _create_sequences(X_train_scaled, y_train_scaled, seq_len)
    X_val_seq, y_val_seq = _create_sequences(X_val_scaled, y_val_scaled, seq_len)
    X_test_seq, y_test_seq = _create_sequences(X_test_scaled, y_test_scaled, seq_len)

    return (
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
        X_test_seq, y_test_seq,
        df_test[seq_len:].reset_index(drop=True),
        scaler_y
    )


def preprocess_sequences_with_time(df, params):
    """Used by Transformer for both zone and regional models"""
    df['Prev_Num_Fires_Result'] = df[TARGET_COL].shift(params['lag_days'])
    df['month'] = df[DATE_COL].dt.month - 3
    df['day_of_week'] = df[DATE_COL].dt.dayofweek

    df_train, df_val, df_test = _split_data(df)

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    month_train = df_train['month'].values
    dow_train = df_train['day_of_week'].values
    month_val = df_val['month'].values
    dow_val = df_val['day_of_week'].values
    month_test = df_test['month'].values
    dow_test = df_test['day_of_week'].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    seq_len = params['sequence_length']
    X_train_seq, y_train_seq, month_train_seq, dow_train_seq = _create_sequences_with_time(
        X_train_scaled, y_train_scaled, month_train, dow_train, seq_len)
    X_val_seq, y_val_seq, month_val_seq, dow_val_seq = _create_sequences_with_time(
        X_val_scaled, y_val_scaled, month_val, dow_val, seq_len)
    X_test_seq, y_test_seq, month_test_seq, dow_test_seq = _create_sequences_with_time(
        X_test_scaled, y_test_scaled, month_test, dow_test, seq_len)

    return (
        X_train_seq, y_train_seq, month_train_seq, dow_train_seq,
        X_val_seq, y_val_seq, month_val_seq, dow_val_seq,
        X_test_seq, y_test_seq, month_test_seq, dow_test_seq,
        df_test[seq_len:].reset_index(drop=True),
        scaler_y
    )