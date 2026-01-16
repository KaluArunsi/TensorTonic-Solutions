import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    a, b = np.asanyarray(a), np.asanyarray(b)
    epsilon = 1e-8

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    cos_val = np.where((norm_a == 0) or (norm_b == 0), 0.0, dot_product / (max(norm_a * norm_b, epsilon)))

    return float(cos_val)