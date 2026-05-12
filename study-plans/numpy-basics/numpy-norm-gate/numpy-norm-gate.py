import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X, W = np.array(X, dtype='float64'), np.array(W, dtype='float64')

    Y = X @ W #Matmul operation of both matrices
    norms = np.linalg.norm(Y, axis=1) #L2 norm
    gate = (norms >= threshold).astype(np.float64) #Threshold gating

    Z = Y * gate[:, np.newaxis] #reshape via broadcasting, zeroing out < threshold items

    return Z