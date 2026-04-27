import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.array(v, dtype='float64')

    l1 = np.sum(np.abs(v))
    l2 = np.sqrt(np.sum(v ** 2)) #can use np.linalg.norm(v) here too tho
    l_inf = np.max(np.abs(v))

    return [l1, l2, l_inf]