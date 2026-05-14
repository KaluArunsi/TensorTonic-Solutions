import numpy as np

def quantize_and_frame(data, decimals, pad_width):
    """Returns: np.ndarray of shape (3, m+2p, n+2p), stacked rounded, floored, ceiled with zero-padding"""
    data = np.array(data, dtype='float64')

    result = []

    for f in [np.round, np.floor, np.ceil]:
        quantized = f(data) if f != np.round else f(data, decimals)
        padded = np.pad(quantized, pad_width, mode='constant')
        result.append(padded)

    return np.stack(result)