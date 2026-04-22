import numpy as np

def create_filled_array(shape, kind):
    """
    Returns: 2D numpy array of given shape with dtype float64
    """
    if kind == 'zeros':
        kind = 0
    elif kind == 'ones':
        kind = 1

    arr = np.full((shape), kind, dtype='float64')

    return arr