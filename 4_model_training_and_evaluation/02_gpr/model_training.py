# GPR model training used by both zone and regional tuning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def build_and_train_model(X_train, y_train, params):
    # Build and train a Gaussian Process Regression model
    kernel = C(params.get('constant_value', 1.0), (1e-3, 1e3)) * \
             RBF(params.get('length_scale', 1.0), (1e-2, 1e2))

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=params.get('alpha', 1e-2),
        normalize_y=True
    )
    model.fit(X_train, y_train)
    return model