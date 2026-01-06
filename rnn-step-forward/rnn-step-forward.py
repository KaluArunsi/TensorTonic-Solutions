import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t = np.asanyarray(x_t).astype(float)
    h_prev = np.asanyarray(h_prev).astype(float)

    input_val = x_t @ Wx
    hidden_val = h_prev @ Wh

    h_t = np.tanh(input_val + hidden_val + b)

    return h_t
