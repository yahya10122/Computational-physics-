import numpy as np

matrix = np.array([[1, -2, 3],
                   [0, -1, 4],
                   [-2, 2, 0]],dtype=float)
aug = np.array([[-16],
                [11],
                [17]])
matrix1 = np.array([[1, -2, 3],
                   [0, -1, 4],
                   [-2, 2, 0]],dtype=float)
n = 3
A = np.ones((n,n))
I = np.eye(n)

def multiplication(matrix,matrix1):
    n = len(matrix)
    sum1 = np.zeros((n,n))
    for j in range(n):
        for k in range(n):
            for i in range(n):
                sum1[j,k] = (matrix[j,i])*matrix1[i,k] + sum1[j,k]
    return(sum1)

def inverse(matrix):
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
    return(ans)

# making LU 
for k in range(0, n - 1):
    for i in range(k + 1, n):
        factor = matrix[i, k] / matrix[k, k]
        for j in range(k, n):
            matrix[i, j] = matrix[i, j] - factor * matrix[k, j]
            matrix1[i, j] = matrix1[i, j] - factor * matrix1[k, j]
        matrix[i, k] = factor
# forming Lower triangluar matrix 
for p in range(0, n):
    for q in range(0, n):
        if p == q:
            matrix[p, q] = 1
        if p < q:
            matrix[p, q] = 0
# Display lower triangular matrix



matrix = inverse(matrix)
matrix1 = inverse(matrix1)

#multiplying out the two matrix
A = multiplication(matrix1,matrix)
print(A)
        
