def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    count = 0
    fx = a * (x0 ** 2) + (b * x0) + c
    x_new = 0

    while count <= steps:
        if count == 0:
            x_old = x0
        else:
            x_old = x_new
        
        grad = (2 * a * x_old) + b

        x_new = x_old - (lr * grad)
        count += 1
    
    return x_new