# Shared hyperparameter sampling and type conversion utilities
import random
import numpy as np


def sample_params(search_space):
    # Randomly samples one value per hyperparameter from the search space
    params = {}
    for key, value in search_space.items():
        if isinstance(value, list):
            params[key] = random.choice(value)
        elif isinstance(value, tuple) and len(value) == 2:
            params[key] = random.randint(*value)
    return params


def clean_for_python(obj):
    # Recursively converts NumPy types to native Python types for serialization
    if isinstance(obj, dict):
        return {k: clean_for_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_python(v) for v in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj