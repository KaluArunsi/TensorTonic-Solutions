import numpy as np

def pearson_correlation(X):
    """
    Returns: ndarray, the Pearson correlation matrix.
    """
    X = np.array(X, dtype='float64')
    X_T = X.T

    corr = np.corrcoef(X_T)

    return corr
    