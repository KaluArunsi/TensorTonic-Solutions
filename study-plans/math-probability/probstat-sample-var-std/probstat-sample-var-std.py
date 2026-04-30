import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x, dtype='float64')

    var_std = {
        'variance': np.var(x, ddof=1),
        'std_dev': np.std(x, ddof=1)
    }

    return var_std