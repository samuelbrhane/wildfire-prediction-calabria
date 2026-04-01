# XGBoost model training used by both zone and regional tuning
import xgboost as xgb

def build_and_train_model(X_train, y_train, params, seed, X_val=None, y_val=None):
    # Build and train an XGBoost regression model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 6),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 1.0),
        colsample_bytree=params.get('colsample_bytree', 1.0),
        gamma=params.get('gamma', 0),
        reg_alpha=params.get('reg_alpha', 0),
        reg_lambda=params.get('reg_lambda', 1),
        random_state=seed,
        verbosity=0
    )

    eval_set = [(X_val, y_val)] if X_val is not None and y_val is not None else None
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model