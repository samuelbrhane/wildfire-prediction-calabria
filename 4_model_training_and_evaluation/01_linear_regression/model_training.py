# Linear Regression model training used by both zone and regional tuning
from sklearn.linear_model import LinearRegression

def build_and_train_model(X_train, y_train, params):
    model = LinearRegression(fit_intercept=params.get('fit_intercept', True))
    model.fit(X_train, y_train)
    return model