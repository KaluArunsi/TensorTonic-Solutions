import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_true, y_pred = np.asanyarray(y_true), np.asanyarray(y_pred)
    d = delta

    e = np.abs(y_true - y_pred)

    cond_1 = 0.5 * np.square(e)
    cond_2 = d * (e - (0.5 * d))
    
    l = np.where((e <= d), cond_1, cond_2)

    l = np.mean(l)

    return l