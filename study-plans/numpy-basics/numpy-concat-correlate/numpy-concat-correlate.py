import numpy as np

def compare_correlations(a, b):
    """Returns: np.ndarray of shape (3, n, n), stacked correlation matrices"""
    a, b = np.array(a, dtype='float64'), np.array(b, dtype='float64')

    combined = np.vstack([a, b])
    corr_a = np.corrcoef(a.T)
    corr_b = np.corrcoef(b.T)
    corr_combined = np.corrcoef(combined.T)

    result = np.stack([corr_a, corr_b, corr_combined])

    return result