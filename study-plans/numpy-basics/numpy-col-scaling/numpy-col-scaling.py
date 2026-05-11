import numpy as np

def scale_cols(data, weights):
    """Returns: np.ndarray of shape (m, n), each column scaled by corresponding weight"""
    data, weight = np.array(data), np.array(weights)

    scale = data * weights

    return scale