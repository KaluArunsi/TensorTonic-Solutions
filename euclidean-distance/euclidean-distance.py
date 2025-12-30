import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    x, y = np.array(x), np.array(y)
    x, y = np.ndarray.flatten(x), np.ndarray.flatten(y) #so they are the same shape

    xy = sum((x-y)**2)

    d = np.sqrt(xy).astype(float)

    return d