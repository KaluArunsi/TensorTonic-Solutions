import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asanyarray(A)

    if A.ndim < 1 or A.shape[0] != A.shape[1]:
        raise ValueError("Input has to be a 2D matrix, boss.")

    rows = A.shape[0]
    trace = 0
    n = 0

    while n < rows:
        trace += A[n, n]
        n += 1

    return trace.item()