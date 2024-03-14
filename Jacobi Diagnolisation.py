import numpy as np

# finding tau,s,c,t
def t_val(A, k, l):
    phi = (-A[k, k] + A[l, l]) / (2 * A[k, l])
    if phi >= 0:
        t = 1 / (abs(phi) + ((phi**2 + 1)) ** 0.5)
    else:
        t = -1 / (abs(phi) + ((phi**2 + 1)) ** 0.5)
    c = 1 / (1 + t**2) ** 0.5
    s = t * c
    tau = s / (1 + c)
    return (t, s, tau, c)


A = np.array([[1, 2**0.5, 2], [2**0.5, 3, 2**0.5], [2, 2**0.5, 1]], dtype=float)
S = np.eye(3)  # Initialize S as the identity matrix

while True:
    # Finding the max value
    n = len(A)
    A1 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j != i:  # Exclude diagonal elements
                A1[i, j] = A[j, i]

    # Break the loop if A1 is an all-zero matrix
    if not np.any(A1):  # This returns False if all elements are 0, thus triggering the break
        break

    # this finds the max value
    max_value = np.max(np.abs(A1))

    # this finds the coordinates of the max value
    max_row, max_col = np.unravel_index(np.argmax(np.abs(A1)), A1.shape)

    # stores it into k and l
    k = max_row
    l = max_col

    # finding A'kk, A'kl,A'lk
    if abs(A[k, l]) > 0.000000001:
        t, s, tau, c = t_val(A, k, l)
        new_S = S.copy()
        new_A = A.copy()
        print(k,l)
        new_A[k, k] = A[k, k] - t * A[k, l]  # finds A'kk
        new_A[l, l] = A[l, l] + t * A[k, l]  # finds A'll
        new_A[k, l] = new_A[l, k] = 0  # equates #A'kl to 0
        
        new_S[k,k] = new_S[l,l] = c
        new_S[k,l] = s
        new_S[l,k] = -s
        R = np.dot(new_S,S)

        # finding the other elements
        for i in range(n):
            if i != k and i != l:
                # finds Ali and Ail where i is only 2 here
                new_A[l, i] = new_A[i, l] = A[l, i] + s * (A[k, i] - tau * A[l, i])

                # finds Aki and Aik wher ei is only 2 here
                new_A[k, i] = new_A[i, k] = A[k, i] - s * (A[l, i] + tau * A[k, i])
            
        # finding eigenvectors
        for i in range(n):
                new_S[i, l] = S[i, l] + s * (S[i, k] - tau * S[i, l])
                new_S[i, k] = S[i, k] - s * (S[i, l] + tau * S[i, k])
        print(new_S)
        S = new_S.copy()
        A = new_A.copy()
        new_S = np.eye(n)
        S = np.copy(R)

    else:
        break

print("Eigenvalues:")
print(np.diag(A))
print("\nEigenvectors:")
print(R)
