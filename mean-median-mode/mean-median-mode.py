import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x = np.array(x)
    x_values, x_count = np.unique(x, return_counts=True)
    value_index = np.argmax(x_count)

    mean = np.mean(x).astype(float)
    median = np.median(x).astype(float)
    mode = x_values[value_index].astype(float)
    
    return mean, median, mode