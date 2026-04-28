import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    data = np.array(data, dtype='float64')

    element_mask = np.where(data > threshold, 1.0, 0.0)
    any_row_mask = np.where(np.any(data > threshold, axis=1, keepdims=True), data, 0.0)
    all_row_mask = np.where(np.all(data > threshold, axis=1, keepdims=True), data, 0.0)

    summary = np.array([element_mask, any_row_mask, all_row_mask], dtype='float64')

    return summary