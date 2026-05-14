import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    data = np.array(data, dtype='float64')

    row_min, row_max = np.min(data, axis=1), np.max(data, axis=1)
    min_col, max_col = np.argmin(data, axis=1), np.argmax(data, axis=1)

    extremes = np.stack([row_max, max_col, row_min, min_col])

    return extremes