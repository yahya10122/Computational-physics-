import numpy as np

matrix = np.array([[-2, 4, 0],
                   [4, -2, 1],
                   [0, -2, 4]],dtype=float)
aug = np.array([[6],
                [3],
                [8]],dtype=float)
n = len(matrix)

U = np.zeros((n,n),dtype=float)
L = np.eye((n),dtype=float)
b = np.zeros((n,1),dtype=float)
c = np.zeros((n-1,1),dtype=float)
a = np.zeros((n-1,1),dtype=float)

# Extract the diagonals
for i in range(n):
    b[i] = matrix[i][i]
    if i < n - 1:
        c[i][0] = matrix[i][i+1]
        a[i][0] = matrix[i+1][i]
        
#make upper and lower triangular matrix
for i in range(n-1): # corrected line
    U[i][i+1] = c[i,0]

# Construct L and U
for i in range(n):
    if i == 0:
        U[i,i] = b[i,0]
    else:
        L[i,i-1] = a[i-1,0]/U[i-1,i-1]
        U[i,i] = b[i,0] - L[i,i-1]*c[i-1,0]
        
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
        sum_val += ans1[j] * U[i, j]
    ans1[i] = (ans[i] - sum_val) / U[i, i]
print(ans1)
