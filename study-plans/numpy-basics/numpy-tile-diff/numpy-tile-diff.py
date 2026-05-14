import numpy as np

def tile_diff(data, reps):
    """Returns: np.ndarray of shape (2, m*reps, n), stacked tiled array and padded differences"""
    data = np.array(data, dtype='float64')

    tiled = np.tile(data, (reps, 1))
    diff = np.diff(tiled, axis=0)
    padded = np.pad(diff, ((0, 1), (0, 0)) , mode='constant')

    result = np.stack([tiled, padded], axis=0)

    return result