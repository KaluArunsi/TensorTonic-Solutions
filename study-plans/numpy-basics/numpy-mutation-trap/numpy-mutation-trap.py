import numpy as np

def original_and_clipped(data, row_idx, lo, hi):
    """
    Returns: 2D ndarray of float64 with shape (2, ncols)
    """
    data = np.array(data, dtype='float64')

    row = data[row_idx].copy()

    condition = [row < lo, row > hi]
    choice = [lo, hi]
    result = np.select(condition, choice, default=row)

    return np.stack([row, result], dtype='float64')