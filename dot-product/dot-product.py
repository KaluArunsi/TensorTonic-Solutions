import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x, y = np.asanyarray(x).astype(float), np.asanyarray(y).astype(float)

    xy = sum(x * y)
    xy = float(xy)

    return xy