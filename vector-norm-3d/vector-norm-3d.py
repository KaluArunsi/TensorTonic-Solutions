import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    
    v = np.asarray(v)
    v_squared = np.square(v)

    scalar_v = np.sum(v_squared)
    vector_v = np.sum(v_squared, axis=-1)

    norm_v = np.where((v_squared.ndim == 1), scalar_v, vector_v)

    norm_v = np.sqrt(norm_v)

    return norm_v