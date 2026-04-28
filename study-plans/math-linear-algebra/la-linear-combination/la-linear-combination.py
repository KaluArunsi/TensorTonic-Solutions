import numpy as np

def linear_combination(vectors, coefficients):
    """
    Returns: float64 array, the weighted sum of vectors.
    """
    vectors = np.array(vectors, dtype='float64')
    coefficients = np.array(coefficients, dtype='float64')

    combo = np.dot(coefficients, vectors)
    combo = combo

    return combo