import numpy as np

def euclidean_distance(x, y):
    """
    Returns: float, the Euclidean distance between x and y.
    """
    x = np.array(x, dtype='float64')
    y = np.array(y, dtype='float64')

    euclid_dist = np.sqrt(np.sum((x-y) ** 2))

    return euclid_dist