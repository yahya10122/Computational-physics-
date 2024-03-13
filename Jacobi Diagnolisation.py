import numpy as np


#finding tau,s,c,t
def t(A, k, l):
    arg = []
    phi = (-A[k, k] + A[l, l]) / (2 * A[k, l])
    if phi>=0:
        t = (1 / (abs(phi) + ((phi**2 + 1))**0.5))
    else:
        t = (-1 / (abs(phi) + ((phi**2 + 1))**0.5))
    c = 1 / (1 + t**2)**0.5
    s = t * c
    tau = s / (1 + c)
    arg.append(phi)
    arg.append(t)
    arg.append(c)
    arg.append(s)
    arg.append(tau)
    return arg

A = np.array([[1, (2)**0.5, 2],
              [2**0.5, 3, 2**0.5],
              [2, 2**0.5, 1]], dtype=float)
arg = []


while True:
    #finding the max value
    n = len(A)
    A1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j > i:
                A1[i, j] = A[j, i]
    max_value = np.max(np.abs(A1))
    max_row, max_col = np.unravel_index(np.argmax(np.abs(A1)), A1.shape)

    k = max_row
    l = max_col
    
    # finding A'kk, A'kl,A'lk
    
    if abs(A[k, l]) > 0.0011:
        arg = t(A, k, l)
        print(arg)
        A[k, k] = A[k, k] - arg[1] * A[k, l]
        A[l, l] = A[l, l] + arg[1] * A[k, l]
        A[k, l] = A[l, k] = 0
        #finding the other elements
        for i in range(n):
            if i != k and i != l:
                A[k, i] = A[i,k] = A[k, i] - arg[3] * (A[l, i] + arg[4] * A[k, i])
                A[l, i] = A[i,l] = A[l, i] + arg[3] * (A[k, i] - arg[4] * A[l, i])
        print(A)
    else:
        break

