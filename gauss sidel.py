import numpy as np

def norm(array1, array2):
    n = 4
    sum = 0
    for i in range(n):
        sum += (array1[i] - array2[i]) ** 2
    ans = sum ** 0.5
    return ans
e=90
# Initialize variables
xo = np.array([[0], [0], [0], [0]], dtype=float)
x1 = np.array([[0], [0], [0], [0]], dtype=float)
B = np.array([[6], [25], [-11], [15]], dtype=float)
A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]], dtype=float)
n = 4

# Perform Gauss-Jacobi iterations:
while e>0.0001:
    for i in range(n):
        sum1 = 0
        for j in range(n):
            if i != j:
                sum1 += A[i, j] * xo[j]
        x1[i] = 1 / A[i, i] * (B[i] - sum1)
    e = norm(xo,x1)
    xo = x1.copy()  # Update xo for the next iteration
    print(xo)

