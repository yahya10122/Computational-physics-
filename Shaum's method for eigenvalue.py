import numpy as np

A = np.array([[2, -1, 0, 0],
              [-1, 2, -1, 0],
              [0, -1, 2, -1],
              [0, 0, -1, 2]], dtype=float)

def sturm(A, L):
    p = [1]
    n = len(A)

    for i in range(n):
        if i == 0:
            p.append((A[i, i] - L) * p[-1])
        else:
            p.append(((A[i, i] - L) * p[-1]) - (A[i, i - 1] ** 2 * p[-2]))

    count = 0
    sign = []

    for i in range(n + 1):
        if i < n:
            if p[i] == 0:
                sign.append(-1 * np.sign(p[i - 1]))
            else:
                sign.append(np.sign(p[i]))
        else:
            sign.append(np.sign(p[n]))
            break

    for i in range(n):
        if sign[i] != sign[i + 1]:
            count += 1

    return count

def Gersch(A):
    n = len(A)
    y = np.zeros(n)  # Initialize y with zeros
    b = np.zeros(n)  # Initialize b with zeros
    c = np.zeros(n)
    d = np.zeros(n)

    # Store diagonal elements of A in y
    for i in range(n):
        y[i] = A[i, i]

    # Calculate the sum of absolute values of off-diagonal elements for each row
    for i in range(n):
        for j in range(n):
            if i != j:
                b[i] += abs(A[i, j])

    for i in range(n):
        c[i] = y[i] - b[i]
        d[i] = y[i] + b[i]

    return min(c), max(d)

LB, UB = Gersch(A)
ub = UB

# bisection
while abs(LB-ub) > 0.0000001:
    if sturm(A,UB)>=1:
        ub=UB
    UB = (LB+ub)/2
    c = sturm(A, UB)
    if c == 0:
        LB=UB

    print(UB)

