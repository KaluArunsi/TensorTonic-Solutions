import numpy as np

def create_sequence(start, stop, param, kind):
    """
    Returns: 1D ndarray of float64 values
    """
    arange_arr = np.arange(start, stop, param, dtype='float64')
    linspace_arr = np.linspace(start, stop, int(param), dtype='float64')

    seq_arr = arange_arr if kind == 'arange' else linspace_arr

    return seq_arr