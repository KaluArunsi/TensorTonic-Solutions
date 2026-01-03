import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x, q = np.atleast_1d(x), np.atleast_1d(q)

    output = np.percentile(x, q)

    return output