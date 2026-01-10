import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: scalar, list, or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x = np.asanyarray(x)

    x_erf = np.vectorize(math.erf)

    gelu_x = (0.5 * x) * (1 + x_erf(x / np.sqrt(2)))

    gelu_x = np.asanyarray(gelu_x).astype(float)

    return gelu_x