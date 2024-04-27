import numpy as np

def lu_decomposition(matrix):
    n = matrix.shape[0]
    L = np.eye(n)  # Initialize L as an identity matrix
    U = matrix.copy()  # Initialize U as a copy of the input matrix

    # Compute LU decomposition
    for k in range(n - 1):
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            for j in range(k, n):
                U[i, j] -= factor * U[k, j]

    return L, U

def forward_backward_sub(L, U, b):
    n = len(L)

    # Forward substitution: Solve Ly = b for y
    y = np.zeros_like(b, dtype=np.float64)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        sum_val = 0
        for j in range(i):
            sum_val += L[i, j] * y[j]
        y[i] = (b[i] - sum_val) / L[i, i]

    # Backward substitution: Solve Ux = y for x
    x = np.zeros_like(b, dtype=np.float64)
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n - 2, -1, -1):
        sum_val = 0
        for j in range(i + 1, n):
            sum_val += U[i, j] * x[j]
        x[i] = (y[i] - sum_val) / U[i, i]

    return x

def inverse_power_method(A, x0, tol=1e-6):
    L, U = lu_decomposition(A)
    x = x0.copy()
    eigenvalue = np.inf
    iterations = 0

    while True:
        z = forward_backward_sub(L, U, x)

        # Normalize z
        z /= np.linalg.norm(z)

        # Compute Rayleigh quotient
        new_eigenvalue = np.dot(z.T, np.dot(A, z))

        # Check convergence
        if np.abs(eigenvalue - new_eigenvalue) < tol:
            break

        eigenvalue = new_eigenvalue
        x = z
        iterations += 1

    return eigenvalue, x, iterations

A = np.array([[1, 2, 3],
              [3, 2, 7],
              [1, 8, 2]], dtype=float)

s = 0.2
n = A.shape[0]
A_star = A - s * np.eye(n)
x0 = np.ones(3)

eigenvalue, eigenvector, iterations = inverse_power_method(A_star, x0)

print("Smallest eigenvalue:", eigenvalue + s)
print("Corresponding eigenvector:", eigenvector)
print("Number of iterations:", iterations)