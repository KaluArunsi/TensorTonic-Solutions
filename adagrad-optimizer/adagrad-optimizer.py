import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    """
    w, g, G = np.asanyarray(w), np.asanyarray(g), np.asanyarray(G)

    g_t = G + np.square(g)

    denom = np.sqrt(g_t) + eps

    w_t = w - (lr / denom) * g

    return w_t, g_t