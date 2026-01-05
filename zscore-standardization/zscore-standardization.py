import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    x = np.asanyarray(X).astype(float)
    u = np.mean(x, axis=axis, keepdims=True)
    o = np.std(x, axis=axis, keepdims=True)

    z = (x - u) / (o + eps)

    return z