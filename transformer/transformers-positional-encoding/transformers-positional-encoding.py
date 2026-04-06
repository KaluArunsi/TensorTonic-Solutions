import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    bounds = -(np.log(10000) / d_model)

    div = np.exp(np.arange(0, d_model, 2) * bounds)

    pos_indices = np.arange(seq_length).reshape(-1,1)

    p_e = np.zeros((seq_length, d_model))

    p_e[:, 0::2] = np.sin(pos_indices * div)
    p_e[:, 1::2] = np.cos(pos_indices * div)

    return p_e