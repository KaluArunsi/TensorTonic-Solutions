import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    x = np.atleast_1d(x).astype(float)

    output = np.maximum(0.0, x)
    
    return output