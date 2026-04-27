import numpy as np

def cosine_similarity(a, b):
    """
    Returns: float in [-1, 1], cosine similarity between a and b.
    """
    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float64')

    norm_a = np.sqrt(np.sum(a ** 2))
    norm_b = np.sqrt(np.sum(b ** 2))

    cos_sim = np.where((norm_a == 0) or (norm_b == 0), 0.0, np.dot(a, b) / np.dot(norm_a, norm_b))

    return float(cos_sim)