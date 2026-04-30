import numpy as np

def percentiles(x, q):
    """
    Returns: numpy array of percentile values.
    """
    x = np.array(x, dtype='float64')
    q = list(q)

    pct = np.percentile(x, q)
    '''
    r = (p/100) * (n-1), so if one value it is one
    instance of p, else you do for them all.

    k = math.floor(r)
    f = r-k
    P_p = x[k] + f * (x[k_+1] - x[k])

    that's how to arrive at np.percentile()
    '''
    return pct
