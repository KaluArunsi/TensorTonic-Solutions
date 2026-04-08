import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.array(A)
    i, j = A.shape

    new_array = np.zeros((j, i))

    for x in range(i):
        new_array[:, x] = A[x, :]

    return new_array