import numpy as np

matrix = np.array([[1, 2, 3,4],
                   [5, -10, 6,7],
                   [9,9,1,10],
                   [11,12,13,1]], dtype=float)

n = len(matrix)

aug = np.eye(n)

#gaussian elimination

for k in range(0, n - 1):
    for i in range(k + 1, n):
        factor = matrix[i, k] / matrix[k, k]
        aug[i] = aug[i] - factor * aug[k]
        for j in range(k, n):
            matrix[i, j] = matrix[i, j] - factor * matrix[k, j]

ans = np.zeros((n, n))

#backward substitution

for i in range(n - 1, -1, -1):
    sum_val = 0
    for j in range(i + 1, n):
        sum_val += ans[j] * matrix[i, j]
    ans[i] = (aug[i] - sum_val) / matrix[i, i]

print(ans)




    