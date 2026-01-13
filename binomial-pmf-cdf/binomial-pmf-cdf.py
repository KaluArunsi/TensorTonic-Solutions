import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    n = int(n)
    p = np.where((0 <= p <= 1), p, 0.0)
    k = np.where((0 <= k <= n), k, 0.0)

    p_k = p ** k
    q = (1 - p) ** (n -k) # this operation is also termed Q
    n_k = comb(n, k, exact=False, repetition=False)

    pmf = n_k * p_k * q
    cdf = 0.0
    i = 0

    while i <= k:
        p_i = p ** i
        q_i = (1 - p) ** (n -i) # this operation is also termed Q
        n_i = comb(n, i, exact=False, repetition=False)

        pmf_i = n_i * p_i * q_i

        cdf += pmf_i
        i += 1
    
    pmf, cdf = float(pmf), float(cdf)

    return pmf, cdf