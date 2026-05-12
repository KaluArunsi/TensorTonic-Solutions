import numpy as np

def summarize(data, axis):
    """Returns: np.ndarray of shape (4, k), rows are mean, std, min, max"""    
    data = np.array(data, dtype='float64')

    mean, std, min, max = np.mean(data, axis=axis), np.std(data, ddof=0, axis=axis), np.min(data, axis=axis), np.max(data, axis=axis)

    return np.stack([mean, std, min, max])