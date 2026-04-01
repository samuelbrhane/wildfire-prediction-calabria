# Hyperparameter search spaces for all models
LINEAR_REGRESSION_SEARCH_SPACE = {
    'fire_lag': (7, 30),
    'climate_lag': (3, 7),
    'fit_intercept': [True, False],
}

XGBOOST_SEARCH_SPACE = {
    'fire_lag': (7, 30),
    'climate_lag': (3, 7),
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': (0.01, 0.3),
    'subsample': (0.8, 1.0),
    'colsample_bytree': (0.8, 1.0),
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 5, 10],
}

GPR_SEARCH_SPACE = {
    'fire_lag': (7, 30),
    'climate_lag': (3, 7),
    'constant_value': (0.1, 10.0),
    'length_scale': (1.0, 50.0),
    'alpha': (1e-4, 1e-1),
}

LSTM_SEARCH_SPACE = {
    'lstm_units': [32, 64, 96],
    'batch_size': [16, 32, 64],
    'dropout_rate': (0.1, 0.5),
    'learning_rate': (1e-5, 1e-2),
    'lag_days': [1, 2, 3],
    'num_layers': [1, 2, 3],
    'sequence_length': (7, 30),
    'activation_function': ['relu', 'tanh'],
    'optimizer': ['adam', 'rmsprop'],
}

TRANSFORMER_SEARCH_SPACE = {
    'batch_size': [16, 32],
    'learning_rate': (1e-4, 3e-4),
    'optimizer': ['adam', 'rmsprop'],
    'lag_days': [1, 2, 3],
    'sequence_length': (7, 30),
    'd_model': [64, 128],
    'num_heads': [2, 4],
    'ff_dim': [64, 128],
    'num_layers': [1, 2],
    'dropout_rate': (0.2, 0.3),
    'month_embed_dim': [4, 8],
    'dow_embed_dim': [2, 4],
}