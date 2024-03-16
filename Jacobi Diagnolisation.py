import numpy as np

def jacobi_diagonalization(A):
    n = len(A)
    S = np.eye(n)  # Initialize S as the identity matrix
    #finds phi, tau, c and s and t
    def t_val(A, k, l):
        phi = (-A[k, k] + A[l, l]) / (2 * A[k, l])
        if phi >= 0:
            t = 1 / (abs(phi) + ((phi ** 2 + 1)) ** 0.5)
        else:
            t = -1 / (abs(phi) + ((phi ** 2 + 1)) ** 0.5)
        c = 1 / (1 + t ** 2) ** 0.5
        s = t * c
        tau = s / (1 + c)
        return (t, s, tau, c)

    while True: # a loop so that the matrix gets updated till the tolerance limit is reached
        
        new_A = A.copy()

        # Finding the max value
        A1 = np.zeros((n, n))
        for i in range(n): 
            for j in range(n):
                if j != i:  # Exclude diagonal elements
                    A1[i, j] = A[i, j]

        # Break the loop if A1 is an all-zero matrix
        if not np.any(A1):  # This returns False if all elements are 0, thus triggering the break
            break
        #finds the max value and its coordinates
        k, l = np.unravel_index(np.argmax(np.abs(A1)), A1.shape)

        # Finding A'kk, A'kl, A'lk
        if abs(A[k, l]) > 0.000000001:
            t, s, tau, c = t_val(A, k, l)
            new_S = S.copy()

            new_A[k, k] = A[k, k] - t * A[k, l]  # finds A'kk
            new_A[l, l] = A[l, l] + t * A[k, l]  # finds A'll
            new_A[k, l] = new_A[l, k] = 0  # equates A'kl to 0

            # Finding the other elements
            for i in range(n):
                if i != k and i != l:
                    # finds Ali and Ail where i is only 2 here
                    new_A[l, i] = new_A[i, l] = A[l, i] + s * (A[k, i] - tau * A[l, i])
                    # finds Aki and Aik where i is only 2 here
                    new_A[k, i] = new_A[i, k] = A[k, i] - s * (A[l, i] + tau * A[k, i])

            # Finding eigenvectors
            for i in range(n):
                new_S[i, l] = S[i, l] + s * (S[i, k] - tau * S[i, l])
                new_S[i, k] = S[i, k] - s * (S[i, l] + tau * S[i, k])

            S = new_S.copy()
            A = new_A.copy()
        else:
            break

    eigenvalues = np.diag(A)
    eigenvectors = S

    return eigenvalues, eigenvectors

A = np.array([[1, 2**0.5, 2], [2**0.5, 3, 2**0.5], [2, 2**0.5, 1]], dtype=float)

eigenvalues, eigenvectors = jacobi_diagonalization(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
for ev in eigenvectors:
    print(ev)
