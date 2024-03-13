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

A = np.array([[2, 3, 6],
              [3, 4, 5],
              [6, 5, 9]], dtype=float)
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
    
    if abs(A[k, l]) > 0.001:
        arg = t(A, k, l)
        new_A = A.copy()
        new_A[k, k] = A[k, k] - arg[1] * A[k, l]
        new_A[l, l] = A[l, l] + arg[1] * A[k, l]
        new_A[k, l] = new_A[l, k] = 0
        #finding the other elements
        for i in range(n):
            if i != k and i != l:
                new_A[l, i] = new_A[i,l] = A[l, i] + arg[3] * (A[k, i] - arg[4] * A[l, i])
                new_A[k, i] = new_A[i,k] = A[k, i] - arg[3] * (A[l, i] + arg[4] * A[k, i])
        A = new_A
    else:
        break

print(A)
