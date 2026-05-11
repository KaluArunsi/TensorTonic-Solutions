import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    data, weights = np.array(data), np.array(weights)
    return data * weights[:, np.newaxis]