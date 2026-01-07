import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.asanyarray(x).astype(float)

    x_max = np.max(x, axis = -1, keepdims=True)

    numerator = x - x_max
    numerator = np.exp(numerator)

    denominator = np.sum(numerator, axis = -1, keepdims=True)

    softmax =  numerator / denominator

    return softmax