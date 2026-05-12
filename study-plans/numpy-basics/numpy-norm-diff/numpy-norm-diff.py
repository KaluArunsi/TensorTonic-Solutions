import numpy as np

def norm_diff(a, b, lo, hi):
    """Returns: np.ndarray of absolute differences after clipping and rescaling to [0, 1]"""
    a, b = np.array(a, dtype='float64'), np.array(b, dtype='float64')

    clipped_a, clipped_b = np.clip(a, lo, hi), np.clip(b, lo, hi)

    rescaled_a, rescaled_b = ((clipped_a - lo) / (hi - lo)), ((clipped_b - lo) / (hi - lo))

    num = rescaled_a - rescaled_b
    den = rescaled_a + rescaled_b

    return np.abs(num)