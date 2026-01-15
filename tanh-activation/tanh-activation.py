import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    x = np.asanyarray(x, dtype=float)
    original_shape = x.shape

    z = np.exp(-2 * np.abs(x))
    result = np.sign(x) * (1 - z) / (1 + z)
    
    if original_shape == ():
        return result.reshape(1)
        
    return result