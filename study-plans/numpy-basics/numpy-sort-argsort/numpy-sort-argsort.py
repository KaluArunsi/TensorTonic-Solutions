import numpy as np

def sort_with_indices(data, axis):
    """Returns: np.ndarray of shape (2, m, n), stacked sorted values and sort indices"""
    data = np.array(data, dtype='float64')

    sorted_values = np.sort(data, axis=axis)
    sorted_indices = np.argsort(data, axis=axis)

    return np.stack([sorted_values, sorted_indices])