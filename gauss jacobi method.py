import numpy as np
xo = np.array([[0],
              [0],
              [0],
              [0]],dtype=float)
B = np.array([[6],
              [25],
              [-11],
              [15]],dtype=float)
A = np.array([[10,-1,2,0],
              [-1,11,-1,3],
              [2,-1,10,-1],
              [0,3,-1,8]],dtype=float)
n = 4
for k in range(5):
    for i in range (n):
        sum1 = 0
        for j in range(n):
            if i!=j:
                sum1 = A[i,j]*xo[j] + sum1
        xo[i] = 1/A[i,i] * (B[i] - sum1)
    print(xo)