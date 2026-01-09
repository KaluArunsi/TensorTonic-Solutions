import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w, v, grad = np.asanyarray(w), np.asanyarray(v), np.asanyarray(grad)

    new_v = (momentum * v) + ( lr * grad)
    new_v = np.asanyarray(new_v)

    new_w = w - new_v
    new_w = np.asanyarray(new_w)

    return new_w, new_v