import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X, y = np.array(X), np.array(y)
    d = X.shape[1]
    w = np.zeros(d)
    b = 0.0


    for _ in range(1, epochs+1):
        y_hat = np.array((X @ w) + b)
        
        dw = (2/len(X)) * X.T @ (y_hat - y)
        db = (2/len(X)) * np.sum(y_hat - y)

        w = w - (lr * dw)
        b = b - (lr * db)

    w = np.round(w, decimals=4)
    b = np.round(b, decimals=4)

    return (w, b)