import numpy as np

def multiplication(matrix1, matrix2):
    n = len(matrix1)
    m = len(matrix2[0])  # number of columns in matrix2
    p = len(matrix2)  # number of rows in matrix2
    result = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i, j] += matrix1[i, k] * matrix2[k, j]

    return result

def euclediannorm(matrix1, matrix2):
    n = len(matrix1)
    sum1 = 0

    for i in range(n):
        sum1 = (matrix1[i] - matrix2[i]) ** 2 + sum1

    norm = (sum1) ** 0.5
    return norm

def eigenvalue(A, x):
    Ax = multiplication(A, x)
    eigval = Ax / x  # Element-wise division
    return eigval[0, 0]  # Return the scalar eigenvalue

def calc_dominant_eigenvector(A):
    n = len(A)
    x_newscaled = np.zeros((n, 1))
    e = 100

    # initial guess
    x = np.zeros((n, 1))
    for i in range(n):
        element = float(input(f"Enter the test matrix element at {i} position: "))
        x[i] = element

    # iterative process to calculate dominant eigenvector
    while e > 0.000001:
        x_old = x.copy()
        x1 = np.zeros((n, 1))
        x1 = multiplication(A, x)

        for k in range(n):
            x_newscaled[k] = x1[k, 0] / x1[np.argmin(abs(x1))]
            x_old[k] = x[k, 0] / x[np.argmin(abs(x))]

        e = euclediannorm(x_newscaled, x_old)
        x = x1.copy()

    dominant_eigenvalue = eigenvalue(A, x_newscaled)
    eigenvector = x_newscaled 

    return dominant_eigenvalue, eigenvector

# Example usage
A = np.array([[7, 10, 13],
              [5, 1, 2],
              [1, 7, 1]], dtype=float)

dominant_eigenvalue, eigenvector = calc_dominant_eigenvector(A)

print("Dominant Eigenvalue:")
print(dominant_eigenvalue)

print("Eigenvector:")
print(eigenvector)