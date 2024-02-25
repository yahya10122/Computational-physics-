import numpy as np 

matrix = np.array([[1, 1, -1],
                   [1, 2, 0],
                   [-1, 0, 5]],dtype=float)
aug = np.array([[0],
                [5],
                [14]],dtype=float)

n = 3
L = np.zeros((n,n))

for i in range(n):
    for j in range(i+1):
        sum1 = 0
        for k in range(j):
            sum1 += L[i][k] * L[j][k]
        if i == j:
            L[i][j] = np.sqrt(matrix[i][i] - sum1)
        else:
            L[i][j] = (1.0 / L[j][j] * (matrix[i][j] - sum1))
Lt = np.transpose(L)
print(L)
# forward substitution to solve LY = B
ans = np.zeros((n, 1))
for i in range(0, n):
    sum_val = 0
    for j in range(0, i):
        sum_val += L[i, j] * ans[j]
    ans[i] = (aug[i] - sum_val) / L[i, i]
print(ans)
# backward substituion to solve UX = B
ans1 = np.zeros((n, 1))
for i in range(n - 1, -1, -1):
    sum_val = 0
    for j in range(i + 1, n):
        sum_val += ans1[j] * Lt[i, j]
    ans1[i] = (ans[i] - sum_val) / Lt[i, i]

print(ans1)