import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    v = np.asanyarray(v)

    norm_v = np.linalg.norm(v, axis=-1, keepdims=True)
    norm_v = np.asanyarray(norm_v)

    bar_v = v / (norm_v + 1e-8)
    bar_v = np.asanyarray(bar_v).astype(float)

    v_hat = np.where(norm_v == 0, v, bar_v)

    return v_hat