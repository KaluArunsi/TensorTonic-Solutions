import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x, q = np.array(x), np.array(q)

    #edge_case = np.where()

    output = np.percentile(x, q)

    return output