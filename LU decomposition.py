import numpy as np

matrix = np.array([[-2, 4, -2],
                   [4, -2, 1],
                   [1, -2, 4]],dtype=float)
aug = np.array([[-16],
                [11],
                [17]])
matrix1 = np.array([[-2, 4, -2],
                   [4, -2, 1],
                   [1, -2, 4]],dtype=float)
n = 3

# making LU 
for k in range(0, n - 1):
    for i in range(k + 1, n):
        factor = matrix[i, k] / matrix[k, k]
        for j in range(k, n):
            matrix[i, j] = matrix[i, j] - factor * matrix[k, j]
            matrix1[i, j] = matrix1[i, j] - factor * matrix1[k, j]
        matrix[i, k] = factor
# forming Lower triangluar matrix 
for p in range(0, n):
    for q in range(0, n):
        if p == q:
            matrix[p, q] = 1
        if p < q:
            matrix[p, q] = 0
# Display lower triangular matrix
print(matrix)

# forward substitution to solve LY = B
ans = np.zeros((n, 1))
for i in range(0, n):
    sum_val = 0
    for j in range(0, i):
        sum_val += matrix[i, j] * ans[j]
    ans[i] = (aug[i] - sum_val) / matrix[i, i]
print(ans)
# backward substituion to solve UX = B
ans1 = np.zeros((n, 1))
for i in range(n - 1, -1, -1):
    sum_val = 0
    for j in range(i + 1, n):
        sum_val += ans1[j] * matrix1[i, j]
    ans1[i] = (ans[i] - sum_val) / matrix1[i, i]

print(ans1)

