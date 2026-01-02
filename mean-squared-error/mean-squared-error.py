import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true, y_pred = np.ndarray.flatten(y_true), np.ndarray.flatten(y_pred)

    size_check = (y_pred.size == y_true.size)

    mse = np.mean((y_pred - y_true) ** 2)
    mse = float(mse)

    output = np.where(size_check, mse, None)

    return float(output)
