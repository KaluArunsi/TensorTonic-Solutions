import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x, p = np.asanyarray(x), np.asanyarray(p)

    if np.allclose(np.sum(p), 1) is False:
        raise ValueError(f"All values in P do not sum up to 1")

    xp = np.sum(x*p)

    return xp