import numpy as np

matrix = np.array([[2, -1, 0,0],
                   [-1, 2, -1,0],
                   [0, -1, 2,-1],
                   [0, 0, -1, 2]],dtype=float)

aug = np.array([[1],
                [0],
                [0],
                [1]],dtype=float)

n = len(matrix)
L = np.zeros((n,n),dtype=float)
U = np.eye((n),dtype=float)
b = np.zeros((n),dtype=float)
c = np.zeros((n-1,1),dtype=float)
a = np.zeros((n-1,1),dtype=float)

# Extract the diagonals
for i in range(n):
    b[i] = matrix[i][i]
    if i < n - 1:
        c[i][0] = matrix[i][i+1]
        a[i][0] = matrix[i+1][i]     
        
#make lower triangular matrix
for i in range(n):
    if i > 0:
        L[i][i-1] = a[i-1,0]
        
# Construct L and U

for i in range(n):
    if i == 0:
        L[i,i] = b[i]
    if i==n-1:
       L[n-1,n-1] = b[n-1] - L[n-1,n-2]*U[n-2,n-1]
    else:
        L[i,i] = b[i] - L[i,i-1]*U[i-1,i]
        U[i,i+1] = c[i,0]/L[i,i]

# forward substitution to solve LY = B

ans = np.zeros((n, 1))
for i in range(0, n):
    sum_val = 0
    for j in range(0, i):
        sum_val += L[i, j] * ans[j]
    ans[i] = (aug[i] - sum_val) / L[i, i]

# backward substituion to solve UX = B

ans1 = np.zeros((n, 1))
for i in range(n - 1, -1, -1):
    sum_val = 0
    for j in range(i + 1, n):
        sum_val += ans1[j] * U[i, j]
    ans1[i] = (ans[i] - sum_val) / U[i, i]
    
#print final answers

labels = ["X1", "X2", "X3", "X4"]  
for i in range(len(ans1)):
    print(f"{labels[i]} = ", ans1[i])




      